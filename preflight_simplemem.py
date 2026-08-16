"""Offline dependency preflight for the vendored Omni-SimpleMem adapter."""

from benchmark.simplemem import _load_simplemem_classes


def main() -> None:
    config_cls, orchestrator_cls = _load_simplemem_classes()
    print(
        "simplemem_import_ok "
        f"config={config_cls.__name__} orchestrator={orchestrator_cls.__name__}"
    )


if __name__ == "__main__":
    main()
