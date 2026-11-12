import Foundation
import Darwin

/// Samples ANOTHER thread's call stack from the recorder's thread — how a stalled main thread
/// gets its stack into the log even though nothing on that thread can run to report itself.
///
/// While the target is suspended we do ONLY raw memory reads into a preallocated buffer: no
/// allocation, no locks, no logging (the target may hold the malloc or log lock). Symbolication
/// happens after it is resumed. Frame-pointer walk; arm64 only (Apple's ABI keeps fp chains).
struct StackSampler: @unchecked Sendable {
    let thread: mach_port_t
    let stackHigh: UInt          // highest valid stack address of the target thread
    private let buffer: UnsafeMutablePointer<UInt>
    static let maxFrames = 48

    /// Call ON the thread to be sampled.
    init() {
        thread = pthread_mach_thread_np(pthread_self())
        stackHigh = UInt(bitPattern: pthread_get_stackaddr_np(pthread_self()))
        buffer = .allocate(capacity: Self.maxFrames)
    }

    /// Raw return addresses of the target's stack, innermost first. Empty if unsupported/failed.
    func capture() -> [UInt] {
        #if arch(arm64)
        guard thread_suspend(thread) == KERN_SUCCESS else { return [] }
        var n = 0
        var state = arm_thread_state64_t()
        var count = mach_msg_type_number_t(MemoryLayout<arm_thread_state64_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &state) {
            $0.withMemoryRebound(to: natural_t.self, capacity: Int(count)) {
                thread_get_state(thread, ARM_THREAD_STATE64, $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            let mask: UInt = 0x0000_007F_FFFF_FFFF           // strip pointer-auth bits
            buffer[n] = UInt(state.__pc) & mask; n += 1
            buffer[n] = UInt(state.__lr) & mask; n += 1
            var fp = UInt(state.__fp)
            let sp = UInt(state.__sp)
            while n < Self.maxFrames, fp >= sp, fp % 16 == 0, fp + 16 <= stackHigh {
                let rec = UnsafePointer<UInt>(bitPattern: fp)!
                let next = rec[0], ret = rec[1] & mask
                if ret == 0 { break }
                buffer[n] = ret; n += 1
                if next <= fp { break }
                fp = next
            }
        }
        thread_resume(thread)                                   // resume BEFORE anything allocates
        return Array(UnsafeBufferPointer(start: buffer, count: n))
        #else
        return []
        #endif
    }

    /// `image  symbol +offset` per address (Swift symbols demangled).
    static func symbolicate(_ addrs: [UInt]) -> [String] {
        addrs.map { a in
            var info = Dl_info()
            guard let p = UnsafeRawPointer(bitPattern: a), dladdr(p, &info) != 0 else {
                return String(format: "0x%lx", a)
            }
            let image = info.dli_fname.map { String(cString: $0).split(separator: "/").last.map(String.init) ?? "?" } ?? "?"
            guard let sname = info.dli_sname else { return "\(image) " + String(format: "0x%lx", a) }
            let off = a - UInt(bitPattern: info.dli_saddr)
            return "\(image)  \(demangle(String(cString: sname))) +\(off)"
        }
    }

    private static func demangle(_ s: String) -> String {
        typealias Fn = @convention(c) (UnsafePointer<CChar>?, Int, UnsafeMutablePointer<CChar>?,
                                       UnsafeMutablePointer<Int>?, UInt32) -> UnsafeMutablePointer<CChar>?
        guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "swift_demangle") else { return s }  // RTLD_DEFAULT
        let fn = unsafeBitCast(sym, to: Fn.self)
        guard let out = s.withCString({ fn($0, strlen($0), nil, nil, 0) }) else { return s }
        defer { free(out) }
        return String(cString: out)
    }
}
