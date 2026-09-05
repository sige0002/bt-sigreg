# LeWorldModel upstream

Official repository: https://github.com/lucas-maes/le-wm

Local checkout: `external/le-wm/`

Pinned commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`

The checkout is ignored by the parent repository. Local integration scripts live
under `scripts/`, keeping the upstream source unmodified.

```bash
git clone https://github.com/lucas-maes/le-wm.git external/le-wm
git -C external/le-wm checkout --detach 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
```

Pretrained smoke-test model: `quentinll/lewm-pusht` on Hugging Face, revision
`22b330c28c27ead4bfd1888615af1340e3fe9052`.
