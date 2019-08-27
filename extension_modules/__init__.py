from pathlib import Path
from dolfin import compile_cpp_code
from typing import Dict, Sequence


def cpp_module(
        header_names: Sequence[str],
        include_dir: str = "include",
        verbose: bool = False
) -> "class <'module'>":
    include_path = Path(__file__).parent.resolve() / include_dir

    header_list = []
    for header in header_names:
        with open(include_path / header, "r") as header_handle:
            header_list.append(header_handle.read())

    separator = "\n"*4
    cpp_code = separator.join(header_list)
    if verbose:
        print(cpp_code)

    module = compile_cpp_code(
        cpp_code,
        include_dirs=[]
    )
    assert module is not None
    return module


class ModuleCache:
    """Cache cpp modules."""

    def __init__(self) -> None:
        """Initialise cache as dicts."""
        self._file_cache: Dict[str, Sequence[str]] = {}     # Cache file names for later compilation
        self._module_cache: Dict[str, "module"] = {}        # name: compiled module

    def add_module(
            self,
            name: str,
            files: Sequence[str],
            compile: bool = False,
            verbose: bool = False
    ) -> None:
        """Add fielanames under module `name`.

        Compile and cache module if `compile` is True.
        """
        self._file_cache[name] = files

        if compile:
            self._module_cache = cpp_module(self._file_cache[name], verbose=verbose)

    def get_module(self, name: str, recompile: bool = False, verbose: bool = False) -> "<class 'module'>":
        """Return module `name`.

        (Re)compile the module if `recompile` is True.
        """
        if name not in self._file_cache:
            assert False, "Could not find module {}".format(name)

        if recompile or name not in self._module_cache:
            self._module_cache[name] = cpp_module(self._file_cache[name], verbose=verbose)

        return self._module_cache[name]


_MODULE_CACHE = ModuleCache()
_MODULE_CACHE.add_module("Cressman", ["vectorised_cressman.h"])
_MODULE_CACHE.add_module("LatticeODESolver", ["vectorised_cressman.h"])
_MODULE_CACHE.add_module("Fitzhugh", ["fitzhugh.h"])
_MODULE_CACHE.add_module("Noble", ["noble.h"])


def load_module(name: str, recompile: bool = False, verbose: bool = False) -> "class< 'module'>":
    """Load module `name`."""
    return _MODULE_CACHE.get_module(name, recompile, verbose=verbose)


if __name__ == "__main__":
    # For testing
    module = load_module("forward_euler")
    print(type(module))
