from __future__ import annotations

import csv
import json
import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .client import CarClient
from .protocol import DEFAULT_HOST, DEFAULT_PORT
from .synthesis import SnapshotDataset, SynthesisResult, synthesize_dataset


rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class MimoCarStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MIMO Car Studio")
        self.geometry("1420x880")
        self.minsize(1180, 720)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.client = CarClient()
        self.dataset = SnapshotDataset()
        self.synthesis: SynthesisResult | None = None
        self.telemetry_log: list[dict[str, Any]] = []
        self.recording = tk.BooleanVar(value=True)
        self._connection_results: queue.Queue[tuple[bool, str]] = queue.Queue()
        self._last_plot = 0.0
        self._time_origin: int | None = None
        self._series: dict[str, deque[float]] = {
            key: deque(maxlen=1500)
            for key in ("t", "x", "y", "xr", "yr", "v", "w", "ir", "il", "ur", "ul")
        }
        self._build_style()
        self._build_ui()
        self.after(50, self._poll)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("Status.TLabel", font=("Consolas", 10))
        style.configure("Danger.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=(16, 8))

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill=tk.X)
        ttk.Label(top, text="MIMO Car Studio", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, text="主机").pack(side=tk.LEFT, padx=(24, 4))
        self.host = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(top, textvariable=self.host, width=16).pack(side=tk.LEFT)
        ttk.Label(top, text="端口").pack(side=tk.LEFT, padx=(10, 4))
        self.port = tk.StringVar(value=str(DEFAULT_PORT))
        ttk.Entry(top, textvariable=self.port, width=7).pack(side=tk.LEFT)
        self.connect_button = ttk.Button(top, text="连接", command=self._toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=8)
        self.connection_label = ttk.Label(top, text="未连接", style="Status.TLabel")
        self.connection_label.pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="急停", style="Danger.TButton", command=self._stop).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body, width=440)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)

        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self._build_control_tab()
        self._build_synthesis_tab()
        self._build_config_tab()
        self._build_plot(right)

    def _build_control_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="运行与监视")
        modes = ttk.LabelFrame(tab, text="工作模式", padding=8)
        modes.pack(fill=tk.X)
        ttk.Button(modes, text="只监视", command=lambda: self._start("monitor")).pack(side=tk.LEFT, padx=3)
        ttk.Button(modes, text="采集数据", command=lambda: self._start("collect")).pack(side=tk.LEFT, padx=3)
        ttk.Button(modes, text="理论控制", command=lambda: self._start("theory")).pack(side=tk.LEFT, padx=3)
        ttk.Button(modes, text="位置清零", command=lambda: self._send("zero_pose")).pack(side=tk.LEFT, padx=3)

        manual = ttk.LabelFrame(tab, text="手动电压测试（先架空车轮）", padding=8)
        manual.pack(fill=tk.X, pady=(8, 0))
        self.manual_r = tk.DoubleVar(value=0.0)
        self.manual_l = tk.DoubleVar(value=0.0)
        for row, (label, variable) in enumerate((("右轮 / V", self.manual_r), ("左轮 / V", self.manual_l))):
            ttk.Label(manual, text=label).grid(row=row, column=0, sticky=tk.W)
            ttk.Scale(manual, from_=-3.0, to=3.0, variable=variable, orient=tk.HORIZONTAL, length=245).grid(row=row, column=1, padx=6)
            ttk.Label(manual, textvariable=variable, width=6).grid(row=row, column=2)
        ttk.Button(manual, text="应用并启动", command=self._manual).grid(row=2, column=0, columnspan=3, pady=(6, 0))

        status = ttk.LabelFrame(tab, text="实时状态", padding=8)
        status.pack(fill=tk.X, pady=(8, 0))
        self.status_vars = {key: tk.StringVar(value="-") for key in (
            "模式", "故障", "传感器", "位姿", "速度", "电流", "电压", "循环", "权重"
        )}
        for row, (key, variable) in enumerate(self.status_vars.items()):
            ttk.Label(status, text=key, width=8).grid(row=row, column=0, sticky=tk.W)
            ttk.Label(status, textvariable=variable, style="Status.TLabel").grid(row=row, column=1, sticky=tk.W)

        log_frame = ttk.LabelFrame(tab, text="通信日志", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(log_frame)
        row.pack(fill=tk.X, pady=(5, 0))
        ttk.Checkbutton(row, text="记录遥测", variable=self.recording).pack(side=tk.LEFT)
        ttk.Button(row, text="导出 CSV", command=self._save_telemetry).pack(side=tk.RIGHT)

    def _build_synthesis_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="数据与综合")
        self.sample_label = tk.StringVar(value="已接收快照：0")
        ttk.Label(tab, textvariable=self.sample_label, style="Title.TLabel").pack(anchor=tk.W)
        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=8)
        ttk.Button(row, text="清空", command=self._clear_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="保存 NPZ", command=self._save_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="加载 NPZ", command=self._load_dataset).pack(side=tk.LEFT, padx=2)

        parameters = ttk.LabelFrame(tab, text="论文综合参数", padding=8)
        parameters.pack(fill=tk.X)
        defaults = {
            "kappa2": "1.0", "kappa3": "2.0", "epsilon2": "0.8",
            "epsilon3": "3.0", "dbar2": "0.004", "dbar3": "0.008",
        }
        self.synthesis_vars = {key: tk.StringVar(value=value) for key, value in defaults.items()}
        labels = {
            "kappa2": "kappa2 (> min)", "kappa3": "kappa3 (> min)", "epsilon2": "epsilon2",
            "epsilon3": "epsilon3", "dbar2": "dbar2", "dbar3": "dbar3",
        }
        for index, key in enumerate(defaults):
            row_index, column = divmod(index, 2)
            ttk.Label(parameters, text=labels[key]).grid(row=row_index, column=column * 2, sticky=tk.W, padx=(0, 4))
            ttk.Entry(parameters, textvariable=self.synthesis_vars[key], width=12).grid(row=row_index, column=column * 2 + 1, padx=(0, 12), pady=2)

        actions = ttk.Frame(tab)
        actions.pack(fill=tk.X, pady=8)
        ttk.Button(actions, text="执行矩阵 Algorithm 1", command=self._synthesize).pack(side=tk.LEFT)
        self.upload_button = ttk.Button(actions, text="上传有效权重", command=self._upload_weights, state=tk.DISABLED)
        self.upload_button.pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="保存到小车", command=lambda: self._send("save")).pack(side=tk.LEFT)

        result_frame = ttk.LabelFrame(tab, text="综合校验", padding=8)
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(result_frame, height=18, font=("Consolas", 10), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _build_config_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="参数")
        defaults = {
            "u_max": "9.0", "current_limit": "1.30", "battery_min": "9.6",
            "run_duration": "30.0", "telemetry_period": "0.020",
            "collection_period": "0.010", "collect_samples": "800",
            "collect_scale": "0.55", "tau1": "0.040", "tau2": "0.015",
            "r2": "0.50", "r3": "0.12", "delta2": "0.08", "delta3": "0.10",
            "kx": "4.0", "ky": "12.0", "kth": "5.0",
            "ref_a": "0.40", "ref_b": "0.20", "ref_nu": "0.20",
        }
        self.config_vars = {key: tk.StringVar(value=value) for key, value in defaults.items()}
        for index, (key, variable) in enumerate(self.config_vars.items()):
            row, column = divmod(index, 2)
            ttk.Label(tab, text=key, width=20).grid(row=row, column=column * 2, sticky=tk.W, pady=2)
            ttk.Entry(tab, textvariable=variable, width=14).grid(row=row, column=column * 2 + 1, sticky=tk.W, padx=(0, 14))
        ttk.Button(tab, text="下发参数", command=self._upload_config).grid(row=11, column=0, pady=12, sticky=tk.W)
        ttk.Button(tab, text="读取小车信息", command=lambda: self._send("hello")).grid(row=11, column=1, pady=12, sticky=tk.W)
        ttk.Button(tab, text="校准 IMU", command=lambda: self._send("calibrate_imu")).grid(row=11, column=2, pady=12, sticky=tk.W)
        ttk.Label(tab, text="配置和校准只能在电机停止时执行。", foreground="#555555").grid(row=12, column=0, columnspan=4, sticky=tk.W)

    def _build_plot(self, parent: ttk.Frame) -> None:
        self.figure = Figure(
            figsize=(9, 7), dpi=100, constrained_layout=True, facecolor="#f4f6f8"
        )
        self.axes = self.figure.subplots(2, 2)
        for axis in self.axes.flat:
            axis.set_visible(False)
        self._plot_active = False
        self._plot_placeholder = self.figure.text(
            0.5,
            0.55,
            "等待小车数据",
            ha="center",
            va="center",
            fontsize=22,
            color="#263238",
            weight="semibold",
        )
        self._plot_hint = self.figure.text(
            0.5,
            0.48,
            "连接 192.168.4.1:8888 后，实时曲线将在这里显示",
            ha="center",
            va="center",
            fontsize=11,
            color="#607d8b",
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(background="#f4f6f8", highlightthickness=0)
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{stamp}] {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _toggle_connection(self) -> None:
        if self.client.connected:
            self.client.disconnect()
            self.connection_label.configure(text="未连接")
            self.connect_button.configure(text="连接")
            return
        try:
            port = int(self.port.get())
        except ValueError:
            messagebox.showerror("端口错误", "端口必须是整数。")
            return
        self.connect_button.configure(state=tk.DISABLED)
        self.connection_label.configure(text="连接中...")
        threading.Thread(target=self._connect_worker, args=(self.host.get().strip(), port), daemon=True).start()

    def _connect_worker(self, host: str, port: int) -> None:
        try:
            self.client.connect(host, port)
            self._connection_results.put((True, f"已连接 {host}:{port}"))
        except Exception as exc:
            self._connection_results.put((False, str(exc)))

    def _send(self, command: str, **parameters: Any) -> None:
        try:
            sequence = self.client.send(command, **parameters)
            if command != "hello":
                self._log(f"TX #{sequence} {command}")
        except Exception as exc:
            self._log(f"发送失败：{exc}")

    def _stop(self) -> None:
        self.manual_r.set(0.0)
        self.manual_l.set(0.0)
        self._send("stop")

    def _start(self, mode: str) -> None:
        self._send("start", mode=mode)

    def _manual(self) -> None:
        self._send("manual", u_r=float(self.manual_r.get()), u_l=float(self.manual_l.get()))
        self._send("start", mode="manual")

    def _clear_dataset(self) -> None:
        self.dataset.clear()
        self.synthesis = None
        self.upload_button.configure(state=tk.DISABLED)
        self.sample_label.set("已接收快照：0")
        self._set_result("")

    def _save_dataset(self) -> None:
        if not len(self.dataset):
            messagebox.showinfo("没有数据", "当前没有可保存的快照。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy dataset", "*.npz")])
        if path:
            self.dataset.save(path)
            self._log(f"数据集已保存：{path}")

    def _load_dataset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NumPy dataset", "*.npz")])
        if path:
            try:
                self.dataset = SnapshotDataset.load(path)
                self.sample_label.set(f"已接收快照：{len(self.dataset)}")
                self._log(f"数据集已加载：{path}")
            except Exception as exc:
                messagebox.showerror("加载失败", str(exc))

    def _synthesize(self) -> None:
        try:
            values = {key: float(variable.get()) for key, variable in self.synthesis_vars.items()}
            self.synthesis = synthesize_dataset(self.dataset, **values)
        except Exception as exc:
            self.synthesis = None
            self.upload_button.configure(state=tk.DISABLED)
            messagebox.showerror("综合失败", str(exc))
            return
        result = self.synthesis
        lines = [f"samples = {result.samples}", ""]
        for name, block in (("block 2", result.block2), ("block 3", result.block3)):
            lines.extend([
                name,
                f"  rank             = {block.rank}/{block.rows}",
                f"  cond(G G^T)      = {block.condition:.6e}",
                f"  matching residual= {block.match_residual:.6e}",
                f"  kappa / min      = {block.kappa:.6e} / {block.kappa_min:.6e}",
                f"  sigma_min(N)-chi = {block.spectral_margin:.6e}",
                f"  Route-II cert    = {block.xi_max:.6e}",
                f"  ||W||_F          = {block.weight_norm:.6e}",
                "",
            ])
        lines.append("PASS：允许上传" if result.valid else "FAIL：不得上传，请重新采集或调整有理论依据的参数")
        self._set_result("\n".join(lines))
        self.upload_button.configure(state=tk.NORMAL if result.valid else tk.DISABLED)

    def _upload_weights(self) -> None:
        if self.synthesis is None or not self.synthesis.valid:
            messagebox.showerror("权重无效", "必须先通过 Route II 的全部论文条件。")
            return
        self._send("set_weights", **self.synthesis.upload_payload())

    def _upload_config(self) -> None:
        try:
            payload: dict[str, float | int] = {key: float(value.get()) for key, value in self.config_vars.items()}
            payload["collect_samples"] = int(payload["collect_samples"])
        except ValueError:
            messagebox.showerror("参数错误", "所有参数必须是数值。")
            return
        self._send("configure", **payload)

    def _set_result(self, text: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.configure(state=tk.DISABLED)

    def _save_telemetry(self) -> None:
        if not self.telemetry_log:
            messagebox.showinfo("没有数据", "尚未记录遥测数据。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        fields = ["t_us", "mode", "fault", "armed", "x", "y", "theta", "v", "omega", "i_r", "i_l", "u_r", "u_l", "loop_us"]
        with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.telemetry_log)
        self._log(f"遥测已导出：{path}")

    def _handle_hello(self, message: dict[str, Any]) -> None:
        config = message.get("config", {})
        for key, variable in self.config_vars.items():
            if key in config:
                variable.set(str(config[key]))
        self.status_vars["权重"].set("有效" if message.get("weights_valid") else "未加载")

    def _handle_telemetry(self, message: dict[str, Any]) -> None:
        sensors = message.get("s", {})
        control = message.get("c", {})
        pose = sensors.get("pose", [0.0, 0.0, 0.0])
        velocity = sensors.get("velocity", [0.0, 0.0])
        current = sensors.get("current", [0.0, 0.0])
        voltage = control.get("u", [0.0, 0.0])
        reference = control.get("reference", [0.0, 0.0, 0.0])
        bus = sensors.get("bus_voltage", [0.0, 0.0, 0.0])
        self.status_vars["模式"].set(f"{message.get('mode', '-')}  armed={message.get('armed', False)}")
        self.status_vars["故障"].set(str(message.get("fault", "-")))
        self.status_vars["传感器"].set(f"INA={sensors.get('ina_ok')}  IMU={sensors.get('imu_ok')}")
        self.status_vars["位姿"].set(f"x={pose[0]:+.3f}  y={pose[1]:+.3f}  th={pose[2]:+.3f}")
        self.status_vars["速度"].set(f"v={velocity[0]:+.3f}  w={velocity[1]:+.3f}")
        self.status_vars["电流"].set(f"R={current[0]:+.3f}  L={current[1]:+.3f} A")
        self.status_vars["电压"].set(f"R={voltage[0]:+.2f}  L={voltage[1]:+.2f} V  bus={max(bus):.2f}")
        self.status_vars["循环"].set(f"{message.get('loop_us', 0)} us  dropped={message.get('dropped', 0)}")
        self.status_vars["权重"].set("有效" if message.get("weights_valid") else "未加载")
        timestamp = int(message.get("t_us", 0))
        if self._time_origin is None:
            self._time_origin = timestamp
        t = (timestamp - self._time_origin) * 1.0e-6
        for key, value in (
            ("t", t), ("x", pose[0]), ("y", pose[1]), ("xr", reference[0]), ("yr", reference[1]),
            ("v", velocity[0]), ("w", velocity[1]), ("ir", current[0]), ("il", current[1]),
            ("ur", voltage[0]), ("ul", voltage[1]),
        ):
            self._series[key].append(float(value))
        if self.recording.get():
            self.telemetry_log.append({
                "t_us": timestamp, "mode": message.get("mode"), "fault": message.get("fault"),
                "armed": message.get("armed"), "x": pose[0], "y": pose[1], "theta": pose[2],
                "v": velocity[0], "omega": velocity[1], "i_r": current[0], "i_l": current[1],
                "u_r": voltage[0], "u_l": voltage[1], "loop_us": message.get("loop_us", 0),
            })
        snapshot = message.get("snapshot")
        if snapshot:
            try:
                self.dataset.append(snapshot)
                self.sample_label.set(f"已接收快照：{len(self.dataset)}")
            except ValueError as exc:
                self._log(f"快照丢弃：{exc}")

    def _redraw(self) -> None:
        if not self._plot_active:
            self._plot_placeholder.set_visible(False)
            self._plot_hint.set_visible(False)
            for axis in self.axes.flat:
                axis.set_visible(True)
            self._plot_active = True
        for axis in self.axes.flat:
            axis.clear()
            axis.set_facecolor("#ffffff")
            axis.grid(True, linestyle=":", linewidth=0.5)
        ax = self.axes[0, 0]
        ax.set_title("平面轨迹")
        ax.plot(self._series["xr"], self._series["yr"], "--", color="#777777", label="reference")
        ax.plot(self._series["x"], self._series["y"], color="#0072B2", label="car")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best")
        t = self._series["t"]
        ax = self.axes[0, 1]
        ax.set_title("速度 [v, omega]")
        ax.plot(t, self._series["v"], label="v")
        ax.plot(t, self._series["w"], label="omega")
        ax.legend(loc="best")
        ax = self.axes[1, 0]
        ax.set_title("分轮电流")
        ax.plot(t, self._series["ir"], label="right")
        ax.plot(t, self._series["il"], label="left")
        ax.set_xlabel("t / s")
        ax.set_ylabel("A")
        ax.legend(loc="best")
        ax = self.axes[1, 1]
        ax.set_title("施加电压")
        ax.plot(t, self._series["ur"], label="right")
        ax.plot(t, self._series["ul"], label="left")
        ax.set_xlabel("t / s")
        ax.set_ylabel("V")
        ax.legend(loc="best")
        self.canvas.draw_idle()

    def _poll(self) -> None:
        try:
            while True:
                ok, message = self._connection_results.get_nowait()
                self.connect_button.configure(state=tk.NORMAL, text="断开" if ok else "连接")
                self.connection_label.configure(text=message if ok else "连接失败")
                self._log(message if ok else f"连接失败：{message}")
        except queue.Empty:
            pass
        for message in self.client.drain():
            kind = message.get("type")
            if kind == "telemetry":
                self._handle_telemetry(message)
            elif kind == "hello":
                self._handle_hello(message)
            elif kind == "connection_closed":
                self.connection_label.configure(text="连接已断开")
                self.connect_button.configure(text="连接", state=tk.NORMAL)
                self._log(f"连接关闭：{message.get('message', '')}")
            else:
                self._log(f"RX #{message.get('seq', '-')} {kind}: {message.get('message', '')}")
        now = time.monotonic()
        if now - self._last_plot > 0.20 and self._series["t"]:
            self._redraw()
            self._last_plot = now
        self.after(50, self._poll)

    def _close(self) -> None:
        try:
            if self.client.connected:
                self.client.send("stop")
                time.sleep(0.05)
        except Exception:
            pass
        self.client.disconnect()
        self.destroy()


def main() -> None:
    MimoCarStudio().mainloop()
