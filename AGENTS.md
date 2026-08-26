# Repository Path Convention

The canonical workspace root is `/data_all/xiangchengzhan/FastWAM`.

`rg` and `grep` print the path spelling supplied to them. For example,
`./scripts/dewo_v2/train.sh` and
`/data_all/xiangchengzhan/FastWAM/scripts/dewo_v2/train.sh` are the same file
when the current directory is the workspace root. Before reporting a path
mismatch, compare `realpath -e` (or the inode), not the textual spelling.

Use this convention in launchers and status reports:

```bash
FASTWAM_ROOT="$(git rev-parse --show-toplevel)"
FASTWAM_ROOT="$(realpath -e -- "${FASTWAM_ROOT}")"
```

The following are deliberately different roots and must be labeled by role:

- `FASTWAM_ROOT`: this repository's source, configs, scripts, and local data.
- `OPEN_REPO` / `FASTWAM_OPEN_REPO`: the sibling `FastWAM-infer-in-DexJoco`
  inference/configuration repository.
- `FASTWAM_PIN`: the pinned source snapshot under
  `third_party/FastWAM_pin_45d8e14`.
- `FASTWAM_UPSTREAM_COPY`: the vendored `third_party/FastWAM` checkout. It is
  currently at the same commit as `FASTWAM_PIN`, but it is a separate copy and
  is not the active import path for this workspace.
- model/checkpoint paths under `/data_all/xiangchengzhan/models/fastwam`.

Do not describe those role-specific roots as inconsistent merely because they
share a machine. For destructive operations, use an explicit canonical
absolute path and verify it with `realpath -e` first.

The active Python source for this workspace is `${FASTWAM_ROOT}/src`; vendored
copies under `third_party/` are reference/pin inputs unless a command names
one explicitly.
