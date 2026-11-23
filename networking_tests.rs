use std::cell::RefCell;

use super::*;
use crate::hle::xgameruntime::xasync::XAsyncGetStatus;

#[test]
fn struct_layouts_match_gdk_abi() {
    assert_eq!(std::mem::size_of::<XNetworkingThumbprint>(), 24);
    assert_eq!(std::mem::size_of::<XNetworkingSecurityInformation>(), 24);
    assert_eq!(std::mem::size_of::<XNetworkingConnectivityHint>(), 16);
}

/// A chain laid out byte by byte as `wincrypt.h` places it on x64, independent of the
/// structs above: `cChain` @ +12, `rgpChain` @ +16, `cLowerQualityChainContext` @ +24;
/// `cElement` @ +12, `rgpElement` @ +16; `pCertContext` @ +8.
#[test]
fn chain_certs_read_at_the_sdk_offsets() {
    #[repr(C, align(8))]
    struct Raw([u8; 64]);
    fn put(raw: &mut Raw, at: usize, bytes: &[u8]) {
        raw.0[at..at + bytes.len()].copy_from_slice(bytes);
    }
    let leaf: CertContext = unsafe { std::mem::zeroed() };
    let root: CertContext = unsafe { std::mem::zeroed() };
    let (leaf, root) = (&leaf as *const CertContext, &root as *const CertContext);

    let mut elements = [Raw([0; 64]), Raw([0; 64])];
    for (e, cert) in elements.iter_mut().zip([leaf, root]) {
        put(e, 0, &24u32.to_le_bytes());
        put(e, 8, &(cert as usize).to_le_bytes());
    }
    let element_ptrs = [&elements[0] as *const Raw, &elements[1] as *const Raw];
    let mut simple = Raw([0; 64]);
    put(&mut simple, 12, &2u32.to_le_bytes());
    put(&mut simple, 16, &(element_ptrs.as_ptr() as usize).to_le_bytes());
    let simple_ptrs = [&simple as *const Raw];
    let mut context = Raw([0; 64]);
    put(&mut context, 12, &1u32.to_le_bytes());
    put(&mut context, 16, &(simple_ptrs.as_ptr() as usize).to_le_bytes());

    unsafe {
        assert_eq!(extract_chain_certs(&context as *const Raw as *const c_void), vec![leaf, root]);
        assert!(extract_chain_certs(std::ptr::null()).is_empty());
        put(&mut context, 12, &0u32.to_le_bytes());
        assert!(extract_chain_certs(&context as *const Raw as *const c_void).is_empty());
    }
}

#[test]
fn udp_port_sync_and_async() {
    let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        let mut port = 0u16;
        assert_eq!(XNetworkingQueryPreferredLocalUdpMultiplayerPort(&mut port), S_OK);
        assert_eq!(port, DEFAULT_UDP_PORT);

        let mut b: XAsyncBlock = std::mem::zeroed();
        assert_eq!(XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync(&mut b), S_OK);
        assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
        let mut async_port = 0u16;
        assert_eq!(XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult(&mut b, &mut async_port), S_OK);
        assert_eq!(async_port, DEFAULT_UDP_PORT);
    }
}

#[test]
fn default_security_info_bytes_zeroes_padding_and_fields() {
    let bytes = default_security_info_bytes();
    assert_eq!(bytes.len(), std::mem::size_of::<XNetworkingSecurityInformation>());
    let off = std::mem::offset_of!(XNetworkingSecurityInformation, enabled_http_security_protocol_flags);
    assert_eq!(&bytes[off..off + 4], &DEFAULT_TLS_FLAGS.to_le_bytes());
    assert!(bytes[..off].iter().all(|&b| b == 0));
    assert!(bytes[off + 4..].iter().all(|&b| b == 0));
}

#[test]
fn security_info_utf8_and_utf16_follow_gdk_arg_order() {
    #[repr(C)]
    struct AlignedBuf {
        _align: [XNetworkingSecurityInformation; 0],
        bytes: [u8; 32],
    }

    let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        // UTF-8
        let mut b: XAsyncBlock = std::mem::zeroed();
        assert_eq!(XNetworkingQuerySecurityInformationForUrlAsync(b"https://example.invalid/\0".as_ptr(), &mut b), S_OK);
        assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
        let mut sz = 0usize;
        assert_eq!(XNetworkingQuerySecurityInformationForUrlAsyncResultSize(&mut b, &mut sz), S_OK);
        assert_eq!(sz, std::mem::size_of::<XNetworkingSecurityInformation>());
        let mut buf = AlignedBuf { _align: [], bytes: [0xAA; 32] };
        let mut used = 0usize;
        let mut ptr: *mut XNetworkingSecurityInformation = std::ptr::null_mut();
        assert_eq!(
            XNetworkingQuerySecurityInformationForUrlAsyncResult(&mut b, buf.bytes.len(), &mut used, buf.bytes.as_mut_ptr(), &mut ptr),
            S_OK
        );
        assert_eq!(used, sz);
        assert_eq!(ptr as *const u8, buf.bytes.as_ptr());
        assert_eq!((*ptr).enabled_http_security_protocol_flags, DEFAULT_TLS_FLAGS);
        assert_eq!((*ptr).thumbprint_count, 0);
        assert!((*ptr).thumbprints.is_null());

        // UTF-16
        let url16 = wide("https://example.invalid/hotfix");
        let mut b2: XAsyncBlock = std::mem::zeroed();
        assert_eq!(XNetworkingQuerySecurityInformationForUrlUtf16Async(url16.as_ptr(), &mut b2), S_OK);
        assert_eq!(XAsyncGetStatus(&mut b2, true), S_OK);
        let mut sz2 = 0usize;
        assert_eq!(XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize(&mut b2, &mut sz2), S_OK);
        assert_eq!(sz2, std::mem::size_of::<XNetworkingSecurityInformation>());
        buf.bytes.fill(0x55);
        let mut ptr2: *mut XNetworkingSecurityInformation = std::ptr::null_mut();
        assert_eq!(
            XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult(&mut b2, buf.bytes.len(), &mut used, buf.bytes.as_mut_ptr(), &mut ptr2),
            S_OK
        );
        assert_eq!(used, sz2);
        assert_eq!(ptr2 as *const u8, buf.bytes.as_ptr());
        assert_eq!((*ptr2).enabled_http_security_protocol_flags, DEFAULT_TLS_FLAGS);
        assert_eq!((*ptr2).thumbprint_count, 0);
        assert!((*ptr2).thumbprints.is_null());
    }
}

#[test]
fn connectivity_hint_and_pin_verification() {
    unsafe {
        let mut hint: XNetworkingConnectivityHint = std::mem::zeroed();
        assert_eq!(XNetworkingGetConnectivityHint(&mut hint), S_OK);
        assert!((1..=4).contains(&hint.connectivity_level));
        assert!((1..=3).contains(&hint.connectivity_cost));
        assert_eq!(hint.iana_interface_type, 6);
        assert!(hint.network_initialized);

        assert_eq!(XNetworkingVerifyServerCertificate(std::ptr::null_mut(), std::ptr::null()), E_INVALIDARG);

        let leaf = vec![0xAAu8; 20];
        let root = vec![0xBBu8; 20];
        let chain = vec![leaf.clone(), root.clone()];
        let pin_ok = XNetworkingThumbprint { thumbprint_type: 2, thumbprint_buffer_byte_count: root.len(), thumbprint_buffer: root.as_ptr() };
        let info_ok = XNetworkingSecurityInformation { enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS, thumbprint_count: 1, thumbprints: &pin_ok };
        assert!(verify_thumbprints(&info_ok, &chain));

        let bad = vec![0xCCu8; 20];
        let pin_bad = XNetworkingThumbprint { thumbprint_type: 0, thumbprint_buffer_byte_count: bad.len(), thumbprint_buffer: bad.as_ptr() };
        let info_bad = XNetworkingSecurityInformation { enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS, thumbprint_count: 1, thumbprints: &pin_bad };
        assert!(!verify_thumbprints(&info_bad, &chain));
    }

    assert_eq!(url_host_and_is_plain_loopback("http://127.0.0.1:3551/account/api/oauth/token"), ("127.0.0.1".into(), true));
    assert_eq!(url_host_and_is_plain_loopback("https://127.0.0.1:3551/secure"), ("127.0.0.1".into(), false));
    assert_eq!(url_host_and_is_plain_loopback("http://[::1]:8080/test"), ("::1".into(), true));
    assert_eq!(url_host_and_is_plain_loopback("https://social.xboxlive.com/users"), ("social.xboxlive.com".into(), false));
    assert_eq!(url_host_and_is_plain_loopback("http://user:pass@127.0.0.1:8080/test"), ("127.0.0.1".into(), true));
    assert_eq!(url_host_and_is_plain_loopback("http://[::1/bad"), ("".into(), false));
    assert_eq!(url_host_and_is_plain_loopback(""), ("".into(), false));
}

#[test]
fn cert_verification_fails_closed_on_malformed_pins_and_zero_win32_errors() {
    assert_eq!(win32_failure_hr(0), crate::util::E_FAIL);
    assert!(win32_failure_hr(87) < 0);
    assert_eq!(policy_status_hr(CERT_E_UNTRUSTEDROOT as u32), CERT_E_UNTRUSTEDROOT);
    assert_eq!(policy_status_hr(87), hresult_from_win32(87));

    let leaf = vec![0xAAu8; 20];
    let issuer = vec![0xBBu8; 20];
    let root = vec![0xCCu8; 20];
    let chain = vec![leaf.clone(), issuer.clone(), root.clone()];

    unsafe {
        // Nonzero thumbprint_count with null pointer must fail closed.
        let null_array = XNetworkingSecurityInformation {
            enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
            thumbprint_count: 1,
            thumbprints: std::ptr::null(),
        };
        assert_eq!(check_thumbprints(&null_array, &chain), PinCheck::InvalidMetadata);
        assert!(!verify_thumbprints(&null_array, &chain));
        assert_eq!(XNetworkingVerifyServerCertificate(1usize as *mut c_void, &null_array), E_INVALIDARG);

        // Empty pin buffer must fail closed even when paired with a matching pin.
        let mixed = [
            XNetworkingThumbprint { thumbprint_type: 0, thumbprint_buffer_byte_count: leaf.len(), thumbprint_buffer: leaf.as_ptr() },
            XNetworkingThumbprint { thumbprint_type: 0, thumbprint_buffer_byte_count: 0, thumbprint_buffer: leaf.as_ptr() },
        ];
        let info_mixed = XNetworkingSecurityInformation {
            enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
            thumbprint_count: mixed.len(),
            thumbprints: mixed.as_ptr(),
        };
        assert_eq!(check_thumbprints(&info_mixed, &chain), PinCheck::InvalidMetadata);
        assert!(!verify_thumbprints(&info_mixed, &chain));

        // Null pin buffer must fail closed.
        let null_buf = XNetworkingThumbprint { thumbprint_type: 0, thumbprint_buffer_byte_count: 20, thumbprint_buffer: std::ptr::null() };
        let info_null_buf = XNetworkingSecurityInformation {
            enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
            thumbprint_count: 1,
            thumbprints: &null_buf,
        };
        assert_eq!(check_thumbprints(&info_null_buf, &chain), PinCheck::InvalidMetadata);
        assert!(!verify_thumbprints(&info_null_buf, &chain));

        // Unknown thumbprint_type must fail closed even if bytes match the leaf cert.
        let unknown_type = XNetworkingThumbprint { thumbprint_type: 3, thumbprint_buffer_byte_count: leaf.len(), thumbprint_buffer: leaf.as_ptr() };
        let info_unknown = XNetworkingSecurityInformation {
            enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
            thumbprint_count: 1,
            thumbprints: &unknown_type,
        };
        assert_eq!(check_thumbprints(&info_unknown, &chain), PinCheck::InvalidMetadata);
        assert!(!verify_thumbprints(&info_unknown, &chain));
        assert_eq!(XNetworkingVerifyServerCertificate(1usize as *mut c_void, &info_unknown), E_INVALIDARG);

        // Valid issuer pin (type 1) matches chain[1].
        let pin_issuer = XNetworkingThumbprint { thumbprint_type: 1, thumbprint_buffer_byte_count: issuer.len(), thumbprint_buffer: issuer.as_ptr() };
        let info_issuer = XNetworkingSecurityInformation {
            enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
            thumbprint_count: 1,
            thumbprints: &pin_issuer,
        };
        assert_eq!(check_thumbprints(&info_issuer, &chain), PinCheck::Matched);
        assert!(verify_thumbprints(&info_issuer, &chain));

        // Overflowing thumbprint_count must fail closed without dereferencing.
        let huge_count = XNetworkingSecurityInformation {
            enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
            thumbprint_count: usize::MAX,
            thumbprints: &pin_issuer,
        };
        assert_eq!(check_thumbprints(&huge_count, &chain), PinCheck::InvalidMetadata);
        assert!(!verify_thumbprints(&huge_count, &chain));

        // Non-20-byte SHA-1 pin length must fail closed even when paired with a valid match.
        let wrong_len = [
            XNetworkingThumbprint { thumbprint_type: 0, thumbprint_buffer_byte_count: leaf.len(), thumbprint_buffer: leaf.as_ptr() },
            XNetworkingThumbprint { thumbprint_type: 0, thumbprint_buffer_byte_count: 16, thumbprint_buffer: leaf.as_ptr() },
        ];
        let info_wrong_len = XNetworkingSecurityInformation {
            enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
            thumbprint_count: wrong_len.len(),
            thumbprints: wrong_len.as_ptr(),
        };
        assert_eq!(check_thumbprints(&info_wrong_len, &chain), PinCheck::InvalidMetadata);
        assert!(!verify_thumbprints(&info_wrong_len, &chain));
    }
}

#[test]
fn raii_guards_drop_chain_before_context() {
    thread_local! {
        static DROP_LOG: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    }

    unsafe extern "system" fn fake_free_ctx(_ctx: *const CertContext) -> i32 {
        DROP_LOG.with(|log| log.borrow_mut().push("context"));
        1
    }

    unsafe extern "system" fn fake_free_chain(_chain: *const c_void) {
        DROP_LOG.with(|log| log.borrow_mut().push("chain"));
    }

    DROP_LOG.with(|log| log.borrow_mut().clear());

    unsafe {
        assert!(CertContextGuard::new(fake_free_ctx, std::ptr::null()).is_none());
        assert!(CertChainGuard::new(fake_free_chain, std::ptr::null()).is_none());
        DROP_LOG.with(|log| assert!(log.borrow().is_empty()));

        let dummy_ctx: CertContext = std::mem::zeroed();
        let dummy_chain = 0u8;
        {
            let ctx = CertContextGuard::new(fake_free_ctx, &dummy_ctx).expect("non-null cert context");
            let chain = CertChainGuard::new(fake_free_chain, &dummy_chain as *const u8 as *const c_void)
                .expect("non-null cert chain");
            assert_eq!(ctx.as_ptr(), &dummy_ctx as *const CertContext);
            assert!(ctx.cert_store().is_null());
            assert_eq!(chain.as_ptr(), &dummy_chain as *const u8 as *const c_void);
            DROP_LOG.with(|log| assert!(log.borrow().is_empty()));
        }

        DROP_LOG.with(|log| {
            assert_eq!(*log.borrow(), vec!["chain", "context"]);
        });
    }
}
