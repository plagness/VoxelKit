import CoreVideo
import CoreMedia

// CVPixelBuffer is safe to pass across concurrency boundaries when used
// with proper retain semantics. Mark as @unchecked Sendable.
extension CVPixelBuffer: @unchecked @retroactive Sendable {}
