from benchmark_models.bsarec import BSARecModel
from benchmark_models.gru4rec import GRU4RecModel
from benchmark_models.sasrec import SASRecModel
from benchmark_models.bert4rec import BERT4RecModel
from benchmark_models.fmlprec import FMLPRecModel
from benchmark_models.duorec import DuoRecModel
from benchmark_models.fearec import FEARecModel
from benchmark_models.wearec import WEARecModel

MODEL_DICT = {
    "bsarec": BSARecModel,
    "gru4rec": GRU4RecModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
    "fmlprec": FMLPRecModel,
    "duorec": DuoRecModel,
    "fearec": FEARecModel,
    "wearec": WEARecModel,
}