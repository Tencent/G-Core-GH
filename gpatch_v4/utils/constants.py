import enum


class GenerateStopReason(enum.Enum):
    FINISH = enum.auto()
    ABORT = enum.auto()
    MAX_LENGTH = enum.auto()
    MAX_GEN_LENGTH = enum.auto()
