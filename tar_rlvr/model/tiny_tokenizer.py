"""스모크용 바이트 토크나이저. Tunix TokenizerAdapter 가 요구하는 인터페이스:
encode / decode / bos_id() / eos_id() / pad_id()  (id 들은 메서드!)
0,1,2 = pad/eos/bos, 3..258 = 바이트(+3), 259.. = <I{k}>"""
import re


class ByteTokenizer:
    OFF = 3  # 바이트 오프셋

    def __init__(self, num_visual_tokens: int = 64):
        self.nv = num_visual_tokens
        self.vis0 = self.OFF + 256
        self.vocab_size = self.vis0 + num_visual_tokens

    def pad_id(self): return 0
    def eos_id(self): return 1
    def bos_id(self): return 2

    def encode(self, s, add_bos: bool = False, **kw):
        # Tunix sampler 가 bos 를 직접 붙이므로 기본 add_bos=False. np.bytes_ 로 들어올 수 있음.
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8", "replace")
        s = str(s)
        ids = [self.bos_id()] if add_bos else []
        for part in re.split(r"(<I\d+>)", s):
            m = re.fullmatch(r"<I(\d+)>", part)
            if m:
                ids.append(self.vis0 + int(m.group(1)) % self.nv)
            else:
                ids += [b + self.OFF for b in part.encode("utf-8", "replace")]
        return ids

    def decode(self, ids, **kw):
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i >= self.vis0:
                out.append(buf.decode("utf-8", "replace")); buf = bytearray()
                out.append(f"<I{i - self.vis0}>")
            elif i >= self.OFF:
                buf.append(i - self.OFF)
            elif i == self.eos_id():
                break
        out.append(buf.decode("utf-8", "replace"))
        return "".join(out)
