import json

from mimo_car_studio.protocol import JsonLineDecoder, encode_message


def test_json_line_roundtrip_and_fragmentation() -> None:
    encoded = encode_message({"seq": 7, "cmd": "stop"})
    decoder = JsonLineDecoder()
    assert decoder.feed(encoded[:5]) == []
    messages = decoder.feed(encoded[5:])
    assert messages == [{"seq": 7, "cmd": "stop", "v": 1}]


def test_decoder_accepts_multiple_frames() -> None:
    decoder = JsonLineDecoder()
    messages = decoder.feed(b'{"type":"a"}\n{"type":"b"}\n')
    assert [message["type"] for message in messages] == ["a", "b"]
