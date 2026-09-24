# trials/ — hands-on learning alongside the book

I work through each chapter by implementing the author's code myself, driven by tests
built from the author's notebook (`chNN/01_main-chapter-code/chNN.ipynb`). The goal is
learning, not coverage: keep the suite small and free of noise.

## Layout (one folder per chapter)

```
trials/chNN/
  <module>.py                 # stubs I implement (book's API only)
  test_01_<topic>.py          # numbered in notebook order
  test_02_<topic>.py
  conftest.py                 # shared fixtures (e.g. raw_text, gpt2_tokenizer)
  <data files>                # copied from the author's chapter folder
```

`trials/ch02/` is the reference example.

## Rules for building a chapter's tests

1. **Book API only.** Stubs contain exactly the classes/functions the author defines
   (same names, same signatures), bodies `raise NotImplementedError`. Do not invent helpers.
2. **Expected values come from the author's notebook outputs.** Extract them from the
   `.ipynb` JSON; don't guess.
3. **No passive tests.** Every test must call my code and fail until I implement it.
   Tests that only exercise libraries (torch, tiktoken) or code written in the test itself
   are out — for concept-only sections I write my own tests.
4. **One or two tests per concept.** Test the key output the author shows. No dtype checks,
   parametrized sweeps, edge cases, or regression-style assertions.
5. **Follow notebook flow.** Files numbered `test_01_`, `test_02_`, … in chapter order;
   inside each file, `# --- N.M Section title ---` headers and tests in cell order.
6. **Notebook setup code goes in fixtures**, copied as-is from the notebook (e.g. building
   the vocab), so tests feed it into my code the same way the author does.
7. **A tiny hand-checkable input is allowed** when it makes a mechanism visible (e.g. a
   7-token string for sliding windows).
8. **Randomness:** use the author's `torch.manual_seed(...)` and exact values if the notebook
   shows them; otherwise assert shapes/invariants.
9. **Style:** bare asserts, descriptive test names, minimal comments.

## Before handing over

- Verify the tests against the author's reference implementation in the scratchpad
  (never in `trials/`): all must pass there, and all must fail against my stubs.
- Python env: `~/miniconda3/envs/py312_0922_llms-from-scratch/bin/python -m pytest trials/chNN`
- VS Code test discovery is scoped to `trials/` via `.vscode/settings.json`.
- Clean up `__pycache__` after running.
