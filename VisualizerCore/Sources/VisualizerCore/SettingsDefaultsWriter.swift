import Foundation

@MainActor
public protocol DefaultsExportableSettings: AnyObject {
    static var sourceFilePath: String { get }
    func exportText() -> String
}

public extension DefaultsExportableSettings {
    @discardableResult
    func saveDefaultsToSource() throws -> URL {
        try SettingsDefaultsWriter.write(
            filePath: Self.sourceFilePath,
            exportText: exportText()
        )
    }
}

@MainActor
public enum SettingsDefaultsWriter {

    public enum WriteError: Error, LocalizedError {
        case sourceMissing(URL)
        case sandboxBlocked(URL)
        case unparseableExportLine(String)
        case unmatchedKeys([String])
        case malformedColorDisplay(String)

        public var errorDescription: String? {
            switch self {
            case .sourceMissing(let url):
                return "Settings source not found at \(url.path)."
            case .sandboxBlocked(let url):
                return "Sandbox blocked writing to \(url.path). Use a Debug build — Release keeps the sandbox on."
            case .unparseableExportLine(let line):
                return "Couldn't parse export line: \(line)"
            case .unmatchedKeys(let keys):
                return "No `var` declarations found for: \(keys.joined(separator: ", "))"
            case .malformedColorDisplay(let value):
                return "Expected `(r, g, b)` for a Color export, got: \(value)"
            }
        }
    }

    @discardableResult
    public static func write(
        filePath: String = #filePath,
        exportText: String
    ) throws -> URL {
        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WriteError.sourceMissing(url)
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let assignments = try parseExport(exportText)

        var patched = source
        var unmatched: [String] = []
        for (key, displayValue) in assignments {
            guard let found = locateLiteral(named: key, in: patched) else {
                unmatched.append(key)
                continue
            }
            let newLiteral = try literal(forDisplay: displayValue, mirroring: found.literal)
            patched.replaceSubrange(found.range, with: newLiteral)
        }
        if !unmatched.isEmpty {
            throw WriteError.unmatchedKeys(unmatched)
        }

        do {
            try patched.write(to: url, atomically: true, encoding: .utf8)
        } catch let nsError as NSError {
            if nsError.domain == NSCocoaErrorDomain && nsError.code == 513 {
                throw WriteError.sandboxBlocked(url)
            }
            throw nsError
        }
        return url
    }

    // MARK: Export parsing

    private static func parseExport(_ text: String) throws -> [(key: String, value: String)] {
        var out: [(String, String)] = []
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            guard let eq = line.firstIndex(of: "=") else {
                throw WriteError.unparseableExportLine(line)
            }
            let key = line[..<eq].trimmingCharacters(in: .whitespaces)
            let value = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
            if key.isEmpty || value.isEmpty {
                throw WriteError.unparseableExportLine(line)
            }
            out.append((key, value))
        }
        return out
    }

    // MARK: Source location

    private static func locateLiteral(
        named name: String,
        in source: String
    ) -> (range: Range<String.Index>, literal: String)? {
        let pattern = #"\bvar\s+\#(NSRegularExpression.escapedPattern(for: name))\b\s*(?::\s*[^=\n]+?)?\s*="#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let nsRange = NSRange(source.startIndex..., in: source)
        guard let match = regex.firstMatch(in: source, range: nsRange),
              let matchRange = Range(match.range, in: source) else { return nil }

        var literalStart = matchRange.upperBound
        while literalStart < source.endIndex,
              source[literalStart] == " " || source[literalStart] == "\t" {
            literalStart = source.index(after: literalStart)
        }
        let literalEnd = scanLiteralEnd(in: source, from: literalStart)
        let raw = String(source[literalStart..<literalEnd])
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return (literalStart..<literalEnd, trimmed)
    }

    private static func scanLiteralEnd(
        in source: String,
        from index: String.Index
    ) -> String.Index {
        var i = index
        var depth = 0
        while i < source.endIndex {
            let c = source[i]
            if c == "(" || c == "[" || c == "{" {
                depth += 1
            } else if c == ")" || c == "]" || c == "}" {
                if depth == 0 { return i }
                depth -= 1
            } else if c == "\n", depth == 0 {
                return i
            } else if c == "/", depth == 0 {
                let next = source.index(after: i)
                if next < source.endIndex, source[next] == "/" {
                    var end = i
                    while end > index,
                          source[source.index(before: end)] == " "
                            || source[source.index(before: end)] == "\t" {
                        end = source.index(before: end)
                    }
                    return end
                }
            }
            i = source.index(after: i)
        }
        return i
    }

    // MARK: Literal rendering

    private static func literal(
        forDisplay display: String,
        mirroring old: String
    ) throws -> String {
        if old.hasPrefix(".") {
            return "." + display
        }
        if old.hasPrefix("Color(") {
            let inner = display.trimmingCharacters(in: CharacterSet(charactersIn: "() "))
            let parts = inner.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            guard parts.count == 3 else {
                throw WriteError.malformedColorDisplay(display)
            }
            return "Color(red: \(parts[0]), green: \(parts[1]), blue: \(parts[2]))"
        }
        return display
    }
}
