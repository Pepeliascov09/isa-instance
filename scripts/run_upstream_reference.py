"""Reference run of instancespace 0.3.0 through the library's documented path.

It does what the package README shows ("Working with the code"), and nothing
else: read the metadata with ``metadata.from_csv_file``, the options with
``options.from_json_file`` (``InstanceSpaceOptions.default()`` when no options
file is given), ``InstanceSpace(metadata, options).build()`` and
``model.save_to_csv(outdir)``. It imports nothing from ``isaspace``: its output
is the reference that ``python -m isaspace.engine --upstream-compat`` must
reproduce file by file (tests/test_upstream_compat.py).

Needs the .venv-isa (Python 3.12, instancespace 0.3.0):
    .venv-isa/bin/python scripts/run_upstream_reference.py \
        --metadata M.csv --outdir FOLDER [--options options.json]

The output folder must not exist or must be empty, so that no file from an
earlier run (e.g. a footprint that is empty in this one) is mistaken for
output of this run.
"""

import argparse
import sys
from pathlib import Path

from instancespace import InstanceSpace, InstanceSpaceOptions
from instancespace.data import metadata, options


def run_reference(metadata_path, outdir, options_path=None):
    """Build the instance space of `metadata_path` and save_to_csv it to `outdir`."""
    outdir = Path(outdir)
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise ValueError(f"{outdir} exists and is not an empty folder")
    metadata_object = metadata.from_csv_file(metadata_path)
    if metadata_object is None:
        raise ValueError(f"instancespace could not read the metadata {metadata_path}")
    if options_path is None:
        options_object = InstanceSpaceOptions.default()
    else:
        options_object = options.from_json_file(options_path)
        if options_object is None:
            raise ValueError(f"instancespace could not read the options {options_path}")

    instance_space = InstanceSpace(metadata_object, options_object)
    try:
        instance_space.build()
    finally:
        instance_space.close()
    outdir.mkdir(parents=True, exist_ok=True)
    instance_space.model.save_to_csv(outdir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--options", default=None,
                        help="options.json (default: InstanceSpaceOptions.default())")
    args = parser.parse_args(argv)
    run_reference(args.metadata, args.outdir, args.options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
