import Foundation
import Darwin

/// Cheap process / machine vitals the flight recorder prints each heartbeat. Every call is a
/// syscall or a Mach call — no allocation-heavy work, safe on the recorder's own thread.
enum FlightVitals {
    /// Physical footprint in bytes (what Activity Monitor calls "Memory", and what jetsam judges).
    static func footprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    /// Peak footprint over the process lifetime, bytes.
    static func peakFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.ledger_phys_footprint_peak) : 0
    }

    /// Kernel memory-pressure level: normal / warn / critical.
    static func pressure() -> String {
        switch sysctlInt("kern.memorystatus_vm_pressure_level") {
        case 1: return "normal"
        case 2: return "WARN"
        case 4: return "CRITICAL"
        default: return "?"
        }
    }

    /// Percent of memory the kernel considers available (100 = plenty, low = thrashing).
    static func availablePercent() -> Int { sysctlInt("kern.memorystatus_level") }

    /// Swap in use, MB.
    static func swapUsedMB() -> Int {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return -1 }
        return Int(usage.xsu_used / 1_048_576)
    }

    /// Process CPU seconds (user + system) so far; the recorder differences it per tick.
    static func cpuSeconds() -> Double {
        var u = rusage()
        getrusage(RUSAGE_SELF, &u)
        func s(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1_000_000 }
        return s(u.ru_utime) + s(u.ru_stime)
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "SERIOUS"
        case .critical: return "CRITICAL"
        @unknown default: return "?"
        }
    }

    private static func sysctlInt(_ name: String) -> Int {
        var v: Int32 = 0
        var size = MemoryLayout<Int32>.size
        return sysctlbyname(name, &v, &size, nil, 0) == 0 ? Int(v) : -1
    }
}
