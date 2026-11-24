"""그룹(G개 롤아웃) 중 일부만 그림을 강제로 시작시킨다. 나머지는 일반 RLVR 그대로.

  - 강제 = 해당 롤아웃의 프롬프트 끝(assistant 턴 시작)에 "<im_start><S2>" 를 붙인다.
    모델은 거기서부터 시각 토큰 → <im_end> → (이어서 텍스트를 쓸지, EOS 를 낼지) 를 스스로 생성한다.
  - 강제 토큰은 프롬프트에 있으므로 loss 대상이 아니다. 학습되는 건 모델이 생성한 부분뿐.
  - 그림 뒤 EOS 가 나와도 텍스트를 붙여주지 않는다 (continuation 과 함께 쓰지 말 것) → 그대로 0점.
  - 강제 롤아웃은 그룹 내 다른 롤아웃과 같은 advantage 정규화를 받는다. 그림을 그린 쪽이 더 잘 맞히면
    그림 → 텍스트 추론 경로의 확률이 오르고, 강제하지 않은 롤아웃에서 스스로 그림을 시작하는지(spontaneous)가 관찰 대상.

Tunix 는 프롬프트를 G 번 연속 복제한 뒤 _generate_and_compute_advantage 를 부른다. 이 함수를 감싸 각 그룹의
마지막 per_group 개 프롬프트를 바꾼다 (그룹 첫 프롬프트는 그대로 → 디버그 로그의 "prompt (공통)" 은 중립 프롬프트).
"""
from __future__ import annotations

DEFAULT_PREFIX = "<im_start><S2>"     # S2 = 9x9 = 81 토큰 (완결 그림이 768 토큰 예산 안에 충분히 들어감)


def is_forced(prompt, prefix: str = DEFAULT_PREFIX) -> bool:
    p = prompt.decode("utf-8", "replace") if isinstance(prompt, (bytes, bytearray)) else str(prompt)
    return p.endswith(prefix)


def install(learner, num_generations: int, per_group: int, prefix: str = DEFAULT_PREFIX):
    from tunix.rl import rl_cluster as rl_cluster_lib
    if not 0 < per_group < num_generations:
        raise ValueError(f"forced_image.per_group={per_group} 은 1..{num_generations - 1} 이어야 한다 (그룹 안에 비교 대상 필요)")
    orig = learner._generate_and_compute_advantage
    G = num_generations

    def gen_adv(training_input, mode=rl_cluster_lib.Mode.TRAIN):
        if mode == rl_cluster_lib.Mode.TRAIN:
            prompts = [p.decode("utf-8", "replace") if isinstance(p, (bytes, bytearray)) else str(p)
                       for p in training_input["prompts"]]     # np.str_ 배열에 더 긴 문자열을 넣으면 잘리므로 list 로
            if len(prompts) % G:
                raise ValueError(f"프롬프트 {len(prompts)}개가 G={G} 의 배수가 아님 — Tunix 복제 규약이 바뀌었는지 확인")
            for s in range(0, len(prompts), G):
                for j in range(G - per_group, G):
                    prompts[s + j] = prompts[s + j] + prefix
            training_input = dict(training_input, prompts=prompts)
        return orig(training_input, mode)

    learner._generate_and_compute_advantage = gen_adv
    print(f"[forced_image] 그룹 {G}개 중 {per_group}개는 '{prefix}' 로 그림을 강제 시작 (loss 제외, 이어쓰기 없음)")
