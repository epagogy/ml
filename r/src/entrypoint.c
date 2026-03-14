// C entry point for R package 'ml'.
// Conditionally links Rust backend when compiled with -DML_HAS_RUST.
// Without Rust, this is a no-op — .rust_available() returns FALSE
// and all algorithms fall back to CRAN packages.

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#ifdef ML_HAS_RUST
extern void R_init_ml_rust_extendr(DllInfo *dll);
#endif

void R_init_ml(DllInfo *dll) {
#ifdef ML_HAS_RUST
    R_init_ml_rust_extendr(dll);
#else
    R_registerRoutines(dll, NULL, NULL, NULL, NULL);
#endif
    R_useDynamicSymbols(dll, FALSE);
}
