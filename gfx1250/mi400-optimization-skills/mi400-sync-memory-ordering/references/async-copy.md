# Async Copy Coordination
Global-to-LDS async movement can overlap computation on another buffer. Do not consume the destination until copy completion and participating-thread/barrier requirements are satisfied. Test fallback paths separately.
