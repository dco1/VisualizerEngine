import Foundation
import Metal

/// Loads a `.metal` / `.metalsource` file for a **runtime source-string compile**, resolving
/// its local `#include "…"` directives first.
///
/// WHY THIS EXISTS. The SwiftPM CLI does not compile this package's `.metal` sources into a
/// metallib — it only copies them (an Xcode-only build rule). So every SwiftPM-side consumer
/// that needs a real pipeline compiles the shader at runtime with
/// `MTLDevice.makeLibrary(source:options:)`.
///
/// That entry point takes a STRING, not a file. It therefore has no notion of "the directory
/// the source came from", and the Metal front end has no include path to search: a local
/// `#include "IlluminatoramaSecondary.h"` fails with
/// `MTLLibraryErrorDomain Code=3 … 'IlluminatoramaSecondary.h' file not found`, even though
/// `xcrun metal -c` on the very same file (which DOES have the file's directory on its search
/// path) compiles clean. That asymmetry is a trap: `swift build` is clean, `xcrun metal -c` is
/// clean, and the failure only appears when a test actually asks the GPU for the library.
/// It cost a real abort — `Illuminatorama.metal` had never included a local header until the
/// analytic night sky moved into a shared one, and the moment it did,
/// `IlluminatoramaMeshSynthTests` could no longer build a library at all.
///
/// The fix is to give the source string what the file system would have given it: splice each
/// local include's text in, recursively, before handing the result to Metal. Sharing a header
/// between shaders is the *correct* design — one definition of the night sky, one definition of
/// the secondary-ray surface shader — so the loader adapts to it rather than the shaders being
/// bent around the harness.
///
/// Angle-bracket includes (`<metal_stdlib>`) are left alone: those resolve against the Metal
/// standard library, which the front end always has.
public enum MetalSourceLoader {

    /// A local header is spliced in AT MOST ONCE per translation unit — the same guarantee its
    /// `#pragma once` gives a real compile, and what keeps a diamond (two shaders including the
    /// same header, or a header including a header) from redefining every struct in it.
    public static func source(contentsOf url: URL) throws -> String {
        var alreadyIncluded = Set<String>()
        return try expand(url: url, alreadyIncluded: &alreadyIncluded)
    }

    /// Read `url`, resolve its includes, and compile the result. The one call a runtime
    /// source-string consumer should make.
    public static func makeLibrary(device: MTLDevice, contentsOf url: URL) throws -> MTLLibrary {
        try device.makeLibrary(source: source(contentsOf: url), options: nil)
    }

    private static func expand(url: URL, alreadyIncluded: inout Set<String>) throws -> String {
        let text = try String(contentsOf: url, encoding: .utf8)
        let dir = url.deletingLastPathComponent()
        var out = String()
        out.reserveCapacity(text.count)

        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            guard let name = localIncludeName(in: line) else {
                out += line
                out += "\n"
                continue
            }
            let headerURL = dir.appendingPathComponent(name)
            // An include we cannot find is left VERBATIM rather than dropped, so Metal reports
            // the missing header itself instead of failing later with a baffling "unknown type".
            guard FileManager.default.fileExists(atPath: headerURL.path) else {
                out += line
                out += "\n"
                continue
            }
            let key = headerURL.standardizedFileURL.resolvingSymlinksInPath().path
            if alreadyIncluded.contains(key) {
                out += "// (\(name) already inlined)\n"
                continue
            }
            alreadyIncluded.insert(key)
            out += "// ── inlined from \(name) (MetalSourceLoader) ──\n"
            out += try expand(url: headerURL, alreadyIncluded: &alreadyIncluded)
            out += "\n// ── end \(name) ──\n"
        }
        return out
    }

    /// The quoted filename of a local `#include "…"`, or nil for anything else (an
    /// angle-bracket include, code, a comment). Deliberately literal: it matches the
    /// directive only at the start of a line (after optional whitespace), which is how every
    /// shader in this package writes it, so a `#include "…"` mentioned inside a comment or a
    /// string is not spliced.
    private static func localIncludeName(in line: Substring) -> String? {
        let trimmed = line.drop { $0 == " " || $0 == "\t" }
        guard trimmed.hasPrefix("#include \"") || trimmed.hasPrefix("#import \"") else { return nil }
        guard let openQuote = trimmed.firstIndex(of: "\"") else { return nil }
        let afterQuote = trimmed.index(after: openQuote)
        guard let closeQuote = trimmed[afterQuote...].firstIndex(of: "\"") else { return nil }
        let name = String(trimmed[afterQuote..<closeQuote])
        return name.isEmpty ? nil : name
    }
}
