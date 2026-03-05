import Foundation
@preconcurrency import Darwin.Mach
import IOKit

// Capture the page size once at load time to avoid concurrency warnings on vm_kernel_page_size.
private let kPageSize: Double = Double(vm_kernel_page_size)

/// Polls macOS kernel for per-core CPU load, memory pressure, and GPU utilization.
/// Call `start()` to begin polling, `stop()` to end.
final class SystemMonitor: @unchecked Sendable {

    private let interval: TimeInterval
    private var task: Task<Void, Never>?

    // Raw CPU tick snapshots for delta calculation
    private var prevTicks: [[UInt32]] = []  // [core][state]

    var onUpdate: (@Sendable ([Double], Double, Double, Double, Double) -> Void)?
    // callback: (perCoreLods, avgLoad, usedGB, totalGB, gpuUtil)

    init(interval: TimeInterval = 1.0) {
        self.interval = interval
    }

    func start() {
        // First poll initialises prevTicks; second poll 1 s later gives real deltas.
        poll()
        task = Task.detached(priority: .utility) { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64((self?.interval ?? 1.0) * 1_000_000_000))
                self?.poll()
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    // MARK: - Polling

    private func poll() {
        let cores = cpuLoads()
        let avg = cores.isEmpty ? 0 : cores.reduce(0, +) / Double(cores.count)
        let (used, total) = memoryStats()
        let gpu = gpuUtilization()
        onUpdate?(cores, avg, used, total, gpu)
    }

    // MARK: - CPU (Mach host_processor_info delta)

    private func cpuLoads() -> [Double] {
        var numCpus: natural_t = 0
        var infoArray: processor_info_array_t?
        var infoCount: mach_msg_type_number_t = 0

        guard host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO,
                                   &numCpus, &infoArray, &infoCount) == KERN_SUCCESS,
              let info = infoArray else { return [] }

        defer {
            let size = vm_size_t(infoCount) * vm_size_t(MemoryLayout<integer_t>.size)
            vm_deallocate(mach_task_self_, vm_address_t(bitPattern: info), size)
        }

        let statesPerCPU = Int(CPU_STATE_MAX)
        var currentTicks: [[UInt32]] = []
        for i in 0..<Int(numCpus) {
            let base = i * statesPerCPU
            let states = (0..<statesPerCPU).map { UInt32(bitPattern: info[base + $0]) }
            currentTicks.append(states)
        }

        var loads: [Double] = []
        if prevTicks.count == currentTicks.count {
            for i in 0..<currentTicks.count {
                let cur = currentTicks[i]
                let prv = prevTicks[i]
                let dUser   = Double(cur[Int(CPU_STATE_USER)]   - prv[Int(CPU_STATE_USER)])
                let dSys    = Double(cur[Int(CPU_STATE_SYSTEM)] - prv[Int(CPU_STATE_SYSTEM)])
                let dIdle   = Double(cur[Int(CPU_STATE_IDLE)]   - prv[Int(CPU_STATE_IDLE)])
                let dNice   = Double(cur[Int(CPU_STATE_NICE)]   - prv[Int(CPU_STATE_NICE)])
                let total = dUser + dSys + dIdle + dNice
                loads.append(total > 0 ? max(0, min(1, (dUser + dSys + dNice) / total)) : 0)
            }
        } else {
            loads = [Double](repeating: 0, count: Int(numCpus))
        }

        prevTicks = currentTicks
        return loads
    }

    // MARK: - Memory (vm_statistics64)

    private func memoryStats() -> (usedGB: Double, totalGB: Double) {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return (0, 0) }

        let pageSize = kPageSize
        let active     = Double(stats.active_count)   * pageSize
        let wired      = Double(stats.wire_count)     * pageSize
        let compressed = Double(stats.compressor_page_count) * pageSize
        let used = (active + wired + compressed) / 1e9
        let total = Double(ProcessInfo.processInfo.physicalMemory) / 1e9
        return (used, total)
    }

    // MARK: - GPU (IOKit IOAccelerator PerformanceStatistics)

    private func gpuUtilization() -> Double {
        let matching = IOServiceMatching("IOAccelerator")
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else {
            return 0
        }
        defer { IOObjectRelease(iterator) }

        var best: Double = 0
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer { IOObjectRelease(service); service = IOIteratorNext(iterator) }

            var props: Unmanaged<CFMutableDictionary>?
            guard IORegistryEntryCreateCFProperties(service, &props,
                                                    kCFAllocatorDefault, 0) == KERN_SUCCESS,
                  let dict = props?.takeRetainedValue() as? [String: AnyObject],
                  let perfStats = dict["PerformanceStatistics"] as? [String: AnyObject] else { continue }

            // Key varies by GPU driver; try both names
            let util = (perfStats["Device Utilization %"] as? Double)
                    ?? (perfStats["GPU Core Utilization"] as? Double).map { $0 / 100 * 100 }
                    ?? (perfStats["Renderer Utilization"] as? Double)
                    ?? 0
            best = max(best, util)
        }
        return min(1, best / 100)
    }

    // MARK: - Free memory helper (for resource limiter)

    static func freeMemoryGB() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size
        )
        let ok = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard ok == KERN_SUCCESS else { return 4 }
        // free + inactive — macOS uses free pages for file cache, so raw free_count is misleading
        return Double(stats.free_count + stats.inactive_count) * kPageSize / 1e9
    }
}
