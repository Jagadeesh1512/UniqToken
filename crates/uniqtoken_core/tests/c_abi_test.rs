use std::ffi::CString;
use std::ptr;
use uniqtoken_core::c_abi::*;
#[test]
fn test_c_abi_null_ptrs() {
    unsafe {
        let res = uniqtoken_export_gguf_vocab(ptr::null(), ptr::null_mut(), ptr::null_mut());
        assert_eq!(res, UNIQTOKEN_ERR_NULL_PTR);
    }
}
#[test]
fn test_c_abi_nonexistent_file() {
    unsafe {
        let path = CString::new("nonexistent_vocab_path_123.json").unwrap();
        let mut buf: *mut std::os::raw::c_void = ptr::null_mut();
        let mut size: usize = 0;
        let res = uniqtoken_export_gguf_vocab(path.as_ptr(), &mut buf, &mut size);
        assert_eq!(res, UNIQTOKEN_ERR_IO);
    }
}
#[test]
fn test_c_abi_demo_vocab_export() {
    unsafe {
        let path_str = concat!(env!("CARGO_MANIFEST_DIR"), "/demo_vocab.json");
        let path = CString::new(path_str).unwrap();
        let mut buf: *mut std::os::raw::c_void = ptr::null_mut();
        let mut size: usize = 0;
        let res = uniqtoken_export_gguf_vocab(path.as_ptr(), &mut buf, &mut size);
        assert_eq!(res, UNIQTOKEN_OK);
        assert!(size > 100);
        assert!(!buf.is_null());
        let bytes = std::slice::from_raw_parts(buf as *const u8, size);
        assert_eq!(&bytes[0..4], b"GGUF");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 3);
        uniqtoken_free_buffer(buf, size);
    }
}
