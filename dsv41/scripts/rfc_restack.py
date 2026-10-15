#!/usr/bin/env python3
"""Restack perf/v41-mono-rfc as three commits on its base, from the current worktree (not touched):
  1. framework + DeepSeek-V4.1   (no DSpark host-sync / fold, no draft stages)
  2. DSpark host-sync / fold     (the files of 011657aeff, independent of monokernels)
  3. DSpark draft stages         (deepseek_v4_dspark.py hooks + the plugin's DRAFT_SPEC) = the worktree
The old history stays on perf/v41-mono-rfc-hist.

  python3 rfc_restack.py /sgl-workspace/sglang_v41mono [--dry-run]
"""

import os
import subprocess
import sys
import tempfile

tree_dir = sys.argv[1]
dry = "--dry-run" in sys.argv
BASE = "b4822bae03"
AUTHOR = {"GIT_AUTHOR_NAME": "wunhuang", "GIT_AUTHOR_EMAIL": "wunhuang@amd.com",
          "GIT_COMMITTER_NAME": "wunhuang", "GIT_COMMITTER_EMAIL": "wunhuang@amd.com"}
HOST_SYNC = [
    "python/sglang/kernels/ops/speculative/dspark/simulated_bonus.py",
    "python/sglang/srt/managers/overlap_utils.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/speculative/dspark_components/dspark_verify.py",
    "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
    "test/registered/spec/dspark/test_dspark_simulated_bonus.py",
]
ENVIRON = "python/sglang/srt/environ.py"
FOLD_ENV = """    # DSpark: apply the simulated correct_len / bonus inside the folded verify epilogue
    # (cuda/hip graph) instead of falling back to the eager accept + commit path.
    SGLANG_DSPARK_SIMULATE_FOLD = EnvBool(True)
"""
DSPARK = "python/sglang/srt/models/deepseek_v4_dspark.py"
SPEC = "python/sglang/kernels/ops/mono/spec.py"
DSV41_SPEC = "python/sglang/kernels/ops/mono/deepseek_v41_gfx950/spec.py"
DRAFT_REG = """    # the DSpark draft model: its stages as whole layers, attached by the draft's own loader
    "DeepseekV4ForCausalLMDSpark": "sglang.kernels.ops.mono.deepseek_v41_gfx950.spec:DRAFT_SPEC",
"""


def git(*a, env=None, inp=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(["git", "-C", tree_dir, *a], check=True, capture_output=True, text=True,
                          env=e, input=inp).stdout.strip()


def worktree(path):
    with open(os.path.join(tree_dir, path)) as f:
        return f.read()


def at_base(path):
    return git("show", f"{BASE}:{path}")


def cut(s, a, b=""):
    assert s.count(a) == 1, (s.count(a), a[:70])
    return s.replace(a, b)


def without_draft_spec(s):
    s = cut(s, """    checked: bool = (
        True  # check mode compares this model's layers against the module path
    )
""")
    s = cut(s, """
        if not self.checked:
            route = route._replace(checker=None)""")
    i, j = s.index("def draft_build("), s.index("def eligible(")
    s = s[:i] + s[j:]
    i = s.index("DRAFT_SPEC = MonoSpec(")
    j = s.index("MONO_SPEC = MonoSpec(")
    return s[:i] + s[j:]


def blob(text):
    return git("hash-object", "-w", "--stdin", inp=text + ("" if text.endswith("\n") else "\n"))


def tree_with(index_env, files):
    """The worktree's tree with ``files`` (path -> text, or None to drop) replaced."""
    for path, text in files.items():
        if text is None:
            git("update-index", "--force-remove", path, env=index_env)
        else:
            git("update-index", "--add", "--cacheinfo", f"100644,{blob(text)},{path}", env=index_env)
    return git("write-tree", env=index_env)


def main():
    head = git("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory() as td:
        idx = {"GIT_INDEX_FILE": os.path.join(td, "index")}
        git("read-tree", head, env=idx)
        git("add", "-A", env=idx)
        t3 = git("write-tree", env=idx)
        environ = worktree(ENVIRON)
        no_draft = {
            DSPARK: at_base(DSPARK),
            SPEC: cut(worktree(SPEC), DRAFT_REG),
            DSV41_SPEC: without_draft_spec(worktree(DSV41_SPEC)),
        }
        t2 = tree_with(idx, no_draft)
        no_sync = {p: at_base(p) for p in HOST_SYNC}
        no_sync[ENVIRON] = cut(environ, FOLD_ENV)
        t1 = tree_with(idx, no_sync)
    msgs = [
        """[AMD][V4.1] Monokernel plugin and DeepSeek-V4.1 decode layers (RFC #43949)

- kernels/ops/mono: the plugin (spec, guards, runtime, check), the shared HIP device runtime (common_hip,
  bounded waits with fail-stop) and DeepSeek-V4.1 on gfx950: every decoder layer in two launches (K1, K2),
  the indexer layers' compressor and indexer between them, scored by the K2i FP4 scorer
- srt: --enable-monokernel (off by default), the config hook, load_tensor / attach / route / kernel at the
  model's call sites; test env vars SGLANG_TEST_MONO_*""",
        """[AMD][V4.1] DSpark: no per-step host sync in decode

- fold the simulated accept / bonus into the verify epilogue (SGLANG_DSPARK_SIMULATE_FOLD, default on)
- keep the overlap scheduler's step free of the host read of the accept lengths""",
        """[AMD][V4.1] DSpark draft stages on monokernels

- the draft model's stages run whole as the DeepSeek-V4.1 launches (the draft's routing and block window)
- registered as DeepseekV4ForCausalLMDSpark, attached by the draft's own loader""",
    ]
    for t, m in zip((t1, t2, t3), msgs):
        print("tree", t[:10], m.splitlines()[0])
    if dry:
        return
    git("branch", "-f", "perf/v41-mono-rfc-hist", head)
    parent = git("rev-parse", BASE)
    for t, m in zip((t1, t2, t3), msgs):
        parent = git("commit-tree", t, "-p", parent, "-m", m, env=AUTHOR)
        print("commit", parent[:10])
    git("update-ref", "refs/heads/perf/v41-mono-rfc", parent)
    git("reset", "-q")  # index = the new HEAD (= the worktree's tree): nothing staged, nothing lost


main()
