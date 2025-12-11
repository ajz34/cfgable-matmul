fn main() {
    // link openblas
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-lib=dylib=gomp");
}
