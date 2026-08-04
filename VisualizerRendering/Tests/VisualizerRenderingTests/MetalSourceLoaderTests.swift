import XCTest
import Foundation
import Metal
@testable import VisualizerRendering

/// THE GATE FOR RUNTIME SOURCE-STRING SHADER COMPILES.
///
/// SwiftPM does not build this package's `.metal` into a metallib, so every SwiftPM-side
/// consumer compiles a shader at runtime with `makeLibrary(source:)`. A source string has no
/// include path, so a local `#include "…"` in that shader fails with
/// `MTLLibraryErrorDomain Code=3 … file not found` — and NOTHING ELSE CATCHES IT:
/// `swift build` only copies the shaders, and `xcrun metal -c` compiles the file with its own
/// directory on the search path, so both are green while the GPU cannot build the library.
///
/// That gap shipped a real abort: `Illuminatorama.metal` had never included a local header, and
/// the moment the analytic night sky moved into a shared one, `IlluminatoramaMeshSynthTests`
/// could not make a library at all. `MetalSourceLoader` splices local includes in first; this
/// test proves it, and — because it ENUMERATES the Shaders directory rather than naming files —
/// it covers the next shared header the day it is written, with no list to update.
final class MetalSourceLoaderTests: XCTestCase {

    private static func shadersDirectory() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // VisualizerRenderingTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // package root
            .appendingPathComponent("Sources/VisualizerRendering/Shaders")
    }

    /// Every `.metal` in the package that uses a local `#include` must still produce a real
    /// `MTLLibrary` from a SOURCE-STRING compile — the path the SwiftPM test suites take.
    func testEveryShaderWithLocalIncludesCompilesFromASourceString() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device on this host")
        }
        let dir = Self.shadersDirectory()
        let all = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertFalse(all.isEmpty, "no .metal sources found at \(dir.path) — the rig drifted")

        var checked: [String] = []
        for url in all {
            let raw = try String(contentsOf: url, encoding: .utf8)
            // Only the shaders that actually use a local header are in scope: the rest have
            // never needed the loader and some are compiled only by Xcode's build rule.
            guard raw.contains("#include \"") else { continue }
            checked.append(url.lastPathComponent)
            // The spliced text must carry no unresolved local include…
            let expanded = try MetalSourceLoader.source(contentsOf: url)
            XCTAssertFalse(expanded.contains("#include \""),
                           "\(url.lastPathComponent): a local #include survived expansion")
            // …and, the real assertion, Metal must build a library from it.
            XCTAssertNoThrow(
                try MetalSourceLoader.makeLibrary(device: device, contentsOf: url),
                "\(url.lastPathComponent) must compile from a source string "
                + "(a local #include cannot resolve without MetalSourceLoader)")
        }
        XCTAssertFalse(checked.isEmpty,
                       "rig check: no shader in this package uses a local #include, so this gate "
                       + "is measuring nothing — if that is now true, delete it deliberately")
        print("MetalSourceLoader: source-string compile OK for \(checked.joined(separator: ", "))")
    }

    /// A DIAMOND — a shader including two headers that both include a third — splices the
    /// shared header ONCE. That is the `#pragma once` guarantee a real compile gives, and
    /// without it the first shared struct to be reached by two paths redefines itself and the
    /// whole library fails to build. Written against temp files rather than the real shaders
    /// because no shader in the package forms a diamond TODAY: the property must be locked
    /// before the layout that needs it exists, not after it breaks.
    func testASharedHeaderReachedTwiceIsInlinedOnlyOnce() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MetalSourceLoaderTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        func write(_ name: String, _ text: String) throws -> URL {
            let url = tmp.appendingPathComponent(name)
            try text.write(to: url, atomically: true, encoding: .utf8)
            return url
        }
        _ = try write("Shared.h", "#pragma once\nstruct Marker { int x; };\n")
        _ = try write("Left.h", "#pragma once\n#include \"Shared.h\"\n")
        _ = try write("Right.h", "#pragma once\n#include \"Shared.h\"\n")
        let root = try write("Root.metal",
                             "#include <metal_stdlib>\n#include \"Left.h\"\n#include \"Right.h\"\n")

        let expanded = try MetalSourceLoader.source(contentsOf: root)
        let definitions = expanded.components(separatedBy: "struct Marker {").count - 1
        XCTAssertEqual(definitions, 1,
                       "a header reached by two include paths must be spliced exactly once "
                       + "(got \(definitions)) — the once-only splice stands in for #pragma once")
        XCTAssertFalse(expanded.contains("#include \""),
                       "every local include must be resolved")
        XCTAssertTrue(expanded.contains("#include <metal_stdlib>"),
                      "angle-bracket includes are left for the Metal front end to resolve")
    }

    /// An unresolvable local include is left VERBATIM, not silently dropped — so Metal reports
    /// the missing header instead of failing later with a baffling "unknown type name".
    func testAMissingLocalIncludeIsLeftForTheCompilerToReport() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MetalSourceLoaderTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let root = tmp.appendingPathComponent("Root.metal")
        try "#include \"NoSuchHeader.h\"\n".write(to: root, atomically: true, encoding: .utf8)
        let expanded = try MetalSourceLoader.source(contentsOf: root)
        XCTAssertTrue(expanded.contains("#include \"NoSuchHeader.h\""),
                      "a missing include must survive so the compiler names it")
    }
}
