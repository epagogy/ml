//! Thin BLAS dispatch — Accelerate on macOS, matrixmultiply/fallback elsewhere.
//!
//! GEMM: `gemm_ab_t`, `gemm_at_b`, `gram_col_major` — used by KNN, Ridge.
//! GEMV: `gemv_rm`, `gemv_rm_t` — used by Logistic (L-BFGS gradient).

// macOS: link to Accelerate framework for hardware-tuned BLAS.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn cblas_dgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );

    fn cblas_dgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );

    fn cblas_daxpy(
        n: i32,
        alpha: f64,
        x: *const f64,
        incx: i32,
        y: *mut f64,
        incy: i32,
    );

    fn cblas_ddot(
        n: i32,
        x: *const f64,
        incx: i32,
        y: *const f64,
        incy: i32,
    ) -> f64;
}

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

/// C(m×n) = A(m×k) × B^T(n×k)  — row-major, B is transposed.
///
/// Use for dot-product matrices: queries(m×d) × train^T → (m×n).
#[inline]
pub fn gemm_ab_t(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// C(m×n) = A^T(k×m) × B(k×n)  — row-major, A is transposed.
///
/// Use for gram matrices: X^T(p×n) × X(n×p) → (p×p).
/// A is (k×m) row-major, read as A^T (m×k).
/// B is (k×n) row-major.
#[inline]
#[allow(dead_code)]
pub fn gemm_at_b(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            m as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            1,
            m as isize,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// C(p×p) = X^T × X  — column-major input (nalgebra DMatrix layout).
///
/// X is (n_rows × n_cols) stored column-major as a flat slice of length n*p.
/// Output C is p×p column-major.
///
/// BLAS call: C = X^T × X with CblasColMajor, transA='T', transB='N'.
#[cfg(target_os = "macos")]
const CBLAS_COL_MAJOR: i32 = 102;

#[inline]
pub fn gram_col_major(
    x: &[f64],
    c: &mut [f64],
    n: usize,
    p: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_dgemm(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,     // A^T
            CBLAS_NO_TRANS,  // B
            p as i32,        // rows of result
            p as i32,        // cols of result
            n as i32,        // inner dimension
            1.0,
            x.as_ptr(),
            n as i32,        // lda = n (column stride of X)
            x.as_ptr(),
            n as i32,        // ldb = n
            0.0,
            c.as_mut_ptr(),
            p as i32,        // ldc = p
        );
    }

    #[cfg(not(target_os = "macos"))]
    unsafe {
        // matrixmultiply uses row/col strides.
        // X col-major (n×p): row stride = 1, col stride = n.
        // X^T: row stride = n, col stride = 1.
        // Result C col-major (p×p): row stride = 1, col stride = p.
        matrixmultiply::dgemm(
            p,               // m = rows of C
            n,               // k = inner
            p,               // n = cols of C
            1.0,
            x.as_ptr(),      // A = X^T
            n as isize,      // A row stride (= X col stride)
            1,               // A col stride (= X row stride)
            x.as_ptr(),      // B = X
            1,               // B row stride
            n as isize,      // B col stride
            0.0,
            c.as_mut_ptr(),
            1,               // C row stride
            p as isize,      // C col stride
        );
    }
}

/// y_out(p) = X^T × y  — column-major X, dense y.
///
/// X is (n × p) column-major. y is (n,). Result is (p,).
#[inline]
pub fn gemv_col_major_t(
    x: &[f64],
    y: &[f64],
    out: &mut [f64],
    n: usize,
    p: usize,
) {
    // Simple loop — p is small (typically <100), no BLAS needed for this.
    for j in 0..p {
        let col = &x[j * n..(j + 1) * n];
        let mut s = 0.0_f64;
        for i in 0..n {
            s += col[i] * y[i];
        }
        out[j] = s;
    }
}

/// y(n) = A(n×p) × x(p)  — row-major A, dense x.
///
/// On macOS uses Accelerate dgemv; elsewhere uses manual loop.
#[inline]
pub fn gemv_rm(
    a: &[f64],
    x: &[f64],
    y: &mut [f64],
    n: usize,
    p: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_dgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            n as i32,
            p as i32,
            1.0,
            a.as_ptr(),
            p as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Iterator-based: clear aliasing → reliable auto-vectorization.
        for i in 0..n {
            let row = &a[i * p..(i + 1) * p];
            y[i] = row.iter().zip(x[..p].iter()).map(|(a, b)| a * b).sum();
        }
    }
}

/// y(p) = A^T(n×p) × x(n)  — row-major A transposed, dense x.
///
/// On macOS uses Accelerate dgemv; elsewhere uses matrixmultiply dgemm.
#[inline]
pub fn gemv_rm_t(
    a: &[f64],
    x: &[f64],
    y: &mut [f64],
    n: usize,
    p: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_dgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            n as i32,
            p as i32,
            1.0,
            a.as_ptr(),
            p as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        // y = A^T × x: scatter each row's contribution (cache-friendly row access).
        for j in 0..p {
            y[j] = 0.0;
        }
        for i in 0..n {
            let row = &a[i * p..(i + 1) * p];
            let xi = x[i];
            for j in 0..p {
                y[j] += row[j] * xi;
            }
        }
    }
}

/// y += alpha * x  — AXPY (the workhorse of coordinate descent).
///
/// On macOS uses Accelerate cblas_daxpy (NEON-vectorized).
/// Elsewhere uses a simple loop that the compiler auto-vectorizes.
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    if n == 0 {
        return;
    }

    #[cfg(target_os = "macos")]
    unsafe {
        cblas_daxpy(n as i32, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Chunk loop helps auto-vectorizer avoid aliasing concerns.
        let xc = &x[..n];
        let yc = &mut y[..n];
        for (yi, xi) in yc.iter_mut().zip(xc.iter()) {
            *yi += alpha * *xi;
        }
    }
}

/// dot(x, y) — inner product.
///
/// On macOS uses Accelerate cblas_ddot (NEON-vectorized).
/// Elsewhere uses a simple loop that the compiler auto-vectorizes.
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    #[cfg(target_os = "macos")]
    unsafe {
        cblas_ddot(n as i32, x.as_ptr(), 1, y.as_ptr(), 1)
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Iterator zip guarantees no aliasing → reliable auto-vectorization.
        x[..n].iter().zip(y[..n].iter()).map(|(a, b)| a * b).sum()
    }
}
