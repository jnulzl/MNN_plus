# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class GpuStage(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGpuStage(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GpuStage()
        x.Init(buf, n + offset)
        return x

    # GpuStage
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GpuStage
    def Pipeline(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # GpuStage
    def GroupSize(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # GpuStage
    def GroupSizeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # GpuStage
    def GroupSizeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuStage
    def InputIndexes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # GpuStage
    def InputIndexesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # GpuStage
    def InputIndexesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuStage
    def OutputIndexes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # GpuStage
    def OutputIndexesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # GpuStage
    def OutputIndexesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuStage
    def MiddleBuffer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .GpuBuffer import GpuBuffer
            obj = GpuBuffer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GpuStage
    def MiddleBufferLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuStage
    def ConstBuffer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .GpuBuffer import GpuBuffer
            obj = GpuBuffer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GpuStage
    def ConstBufferLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuStage
    def GlobalSizeIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # GpuStage
    def GlobalSizeDivide(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # GpuStage
    def GlobalSizeDivideAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # GpuStage
    def GlobalSizeDivideLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuStage
    def RequireSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def GpuStageStart(builder): builder.StartObject(9)
def GpuStageAddPipeline(builder, pipeline): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(pipeline), 0)
def GpuStageAddGroupSize(builder, groupSize): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(groupSize), 0)
def GpuStageStartGroupSizeVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuStageAddInputIndexes(builder, inputIndexes): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(inputIndexes), 0)
def GpuStageStartInputIndexesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuStageAddOutputIndexes(builder, outputIndexes): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(outputIndexes), 0)
def GpuStageStartOutputIndexesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuStageAddMiddleBuffer(builder, middleBuffer): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(middleBuffer), 0)
def GpuStageStartMiddleBufferVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuStageAddConstBuffer(builder, constBuffer): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(constBuffer), 0)
def GpuStageStartConstBufferVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuStageAddGlobalSizeIndex(builder, globalSizeIndex): builder.PrependInt32Slot(6, globalSizeIndex, 0)
def GpuStageAddGlobalSizeDivide(builder, globalSizeDivide): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(globalSizeDivide), 0)
def GpuStageStartGlobalSizeDivideVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuStageAddRequireSize(builder, requireSize): builder.PrependBoolSlot(8, requireSize, 0)
def GpuStageEnd(builder): return builder.EndObject()
