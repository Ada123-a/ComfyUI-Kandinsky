from typing import List, Type
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


from .nodes import KandinskyLoader, KandinskyTextEncode, EmptyKandinskyLatent, KandinskyImageToVideoLatent, KandinskyPruneFrames, KandinskyVAELoader, KandinskyVAEDecode, KandinskyQwenLoader
from .kandinsky_sampler import KandinskySampler

class KandinskyV5Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> List[Type[io.ComfyNode]]:
        return [
            KandinskyLoader,
            KandinskyTextEncode,
            EmptyKandinskyLatent,
            KandinskyImageToVideoLatent,
            KandinskySampler,
            KandinskyVAELoader,
            KandinskyVAEDecode,
            KandinskyPruneFrames,
            KandinskyQwenLoader,
        ]

async def comfy_entrypoint() -> KandinskyV5Extension:
    return KandinskyV5Extension()