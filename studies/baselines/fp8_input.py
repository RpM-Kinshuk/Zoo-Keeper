"""Explicit decoded-value input for modelDNA; its sampling/decision code is unchanged.

Native modelDNA0.1.0 treats FP8 codes as uint8 and ignores scales. This adapter
reuses Zoo's existing decoder and expands physical reads to whole rows; it is
not native FP8 support. Original element positions and algorithm parameters stay
unchanged. Raw bytes/scales remain prefetched by the published reader.
"""
import math
import numpy as np
from modeldna.io.weights import WeightIndex
from evidence.safetensors_header import decode_plain
from evidence.views import plan_views,decode_view


class DecodedFP8Index(WeightIndex):
    def __init__(self, original, entries, config):
        self.__dict__.update(original.__dict__)
        self.entries,self.config=entries,config
        views,_=plan_views(entries,config)
        self.fp8={v.name:v for v in views if v.decode=='fp8_scaled' and v.geometry=='matrix'}
        auxiliaries={v.sources[1] for v in self.fp8.values()}
        self.tensors={name:t for name,t in self.tensors.items() if name not in auxiliaries}
        for name,entry in entries.items():
            if entry.dtype.startswith('F8_') and name not in self.fp8:
                raise ValueError('unsupported FP8 representation in native-input adapter')

    def _physical(self,name,rows):
        entry=self.entries[name]
        shape=tuple(entry.shape);start=entry.start;end=entry.end
        if rows is not None:
            first,last,step=rows.indices(shape[0])
            if step!=1:raise ValueError('contiguous rows required')
            stride=(entry.end-entry.start)//shape[0]
            start+=first*stride;end=entry.start+last*stride;shape=(last-first,*shape[1:])
        return decode_plain(self._read(entry.file,start,end),entry.dtype,shape)

    def _expanded(self,name,start,count):
        t=self.info(name);start=max(0,min(start,t.numel));count=min(count,t.numel-start)
        cols=t.shape[1];first=start//cols;last=math.ceil((start+count)/cols)
        return start,count,first,last

    def read_tensor(self,name):
        if name not in self.fp8:return super().read_tensor(name)
        return decode_view(self.fp8[name],self._physical,self.config).numpy()

    def read_flat_slice(self,name,start_elem,n_elem):
        if name not in self.fp8:return super().read_flat_slice(name,start_elem,n_elem)
        start,count,first,last=self._expanded(name,start_elem,n_elem)
        if count==0:return np.empty(0,dtype=np.float32)
        decoded=decode_view(self.fp8[name],self._physical,self.config,row_slice=slice(first,last)).numpy().reshape(-1)
        offset=start-first*self.info(name).shape[1]
        return decoded[offset:offset+count]

    def plan_flat_slice(self,name,start_elem,n_elem):
        if name not in self.fp8:return super().plan_flat_slice(name,start_elem,n_elem)
        _,count,first,last=self._expanded(name,start_elem,n_elem);t=self.info(name)
        return [] if count==0 else [(t.start+first*t.shape[1],t.start+last*t.shape[1])]

    def plan_sample(self,name,seed,n_blocks=64,block_len=4096):
        original=WeightIndex.plan_sample(self,name,seed,n_blocks,block_len)
        if name not in self.fp8:return original
        t=self.info(name)
        return [span for a,b in original for span in self.plan_flat_slice(name,a-t.start,b-a)]

    def read_sample(self,name,seed,n_blocks=64,block_len=4096):
        if name not in self.fp8:return super().read_sample(name,seed,n_blocks,block_len)
        # Use the native raw-coordinate plan, not the expanded physical-read plan.
        t=self.info(name);ranges=WeightIndex.plan_sample(self,name,seed,n_blocks,block_len)
        return np.concatenate([self.read_flat_slice(name,a-t.start,b-a) for a,b in ranges])

    def prefetch(self,ranges,gap=None):
        ranges=list(ranges)
        scale_names={view.sources[1] for view in self.fp8.values()}
        ranges.extend((self.entries[n].file,self.entries[n].start,self.entries[n].end) for n in scale_names)
        return super().prefetch(ranges,gap)


def decoded_extractor(source,entries,config,**kwargs):
    """Run the published extractor on logical dense values, excluding scale auxiliaries."""
    from modeldna.fingerprint.extract import FingerprintExtractor
    from modeldna.arch.canonical import canonicalize
    from modeldna.arch.signature import read_signature
    extractor=FingerprintExtractor(source,**kwargs)
    extractor.index=DecodedFP8Index(extractor.index,entries,config)
    extractor.cmap=canonicalize(extractor.index.names)
    extractor.sig=read_signature(source,extractor.index,extractor.cmap)
    return extractor
