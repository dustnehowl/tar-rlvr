"""스모크용 바이트 토크나이저: 0..255 = 바이트, 256.. = <I{k}>, 그리고 <answer> 태그는 그냥 바이트."""
import re


class ByteTokenizer:
    def __init__(self, num_visual_tokens: int = 64):
        self.nv = num_visual_tokens
        self.pad_id, self.eos_id, self.bos_id = 0, 1, 2

    def encode(self, s: str, add_bos=True):
        ids = [self.bos_id] if add_bos else []
        for part in re.split(r"(<I\d+>)", s):
            m = re.fullmatch(r"<I(\d+)>", part)
            ids += [256 + int(m.group(1)) % self.nv] if m else list(part.encode("utf-8", "replace"))
        return ids

    def decode(self, ids):
        out, buf = [], bytearray()
        for i in ids:
            if i >= 256:
                out.append(buf.decode("utf-8", "replace")); buf = bytearray(); out.append(f"<I{i-256}>")
            elif i > 2:
                buf.append(i)
        out.append(buf.decode("utf-8", "replace"))
        return "".join(out)
