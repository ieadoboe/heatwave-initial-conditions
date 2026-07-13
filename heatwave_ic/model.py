"""NeuralGCM checkpoint loading (with the optimizer's decoder patch)."""

import pickle

import gcsfs
import neuralgcm


def _gcs():
    return gcsfs.GCSFileSystem(token="anon")


def load_checkpoint(model_name: str) -> dict:
    """Fetch a NeuralGCM checkpoint dict from gs://neuralgcm/models/."""
    with _gcs().open(f"gs://neuralgcm/models/{model_name}", "rb") as f:
        return pickle.load(f)


def load_model(model_name: str, patch_surface_pressure_decoder: bool | None = None):
    """Load a NeuralGCM PressureLevelModel.

    patch_surface_pressure_decoder: apply the config-string surgery that makes
    the decoder also output surface pressure (needed by the optimizer's saved
    diagnostics). The surgery is the one Tim used for the STOCHASTIC-PRECIP
    checkpoint; it likely needs adjusting for a deterministic checkpoint
    (untested), so the default (None) patches only when 'precip' is in the
    model name.
    """
    ckpt = load_checkpoint(model_name)
    if patch_surface_pressure_decoder is None:
        patch_surface_pressure_decoder = "precip" in model_name
    if not patch_surface_pressure_decoder:
        return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

    new_inputs_to_units_mapping = {
        "u": "meter / second",
        "v": "meter / second",
        "t": "kelvin",
        "z": "m**2 s**-2",
        "sim_time": "dimensionless",
        "tracers": {
            "specific_humidity": "dimensionless",
            "specific_cloud_liquid_water_content": "dimensionless",
            "specific_cloud_ice_water_content": "dimensionless",
        },
        "diagnostics": {"surface_pressure": "kg / (meter s**2)"},
    }
    ckpt["model_config_str"] = "\n".join([
        ckpt["model_config_str"],
        "DimensionalLearnedPrimitiveToWeatherbenchDecoder.inputs_to_units_mapping"
        f" = {new_inputs_to_units_mapping}",
        "DimensionalLearnedPrimitiveToWeatherbenchDecoder.diagnostics_module ="
        " @NodalModelDiagnosticsDecoder",
        "StochasticPhysicsParameterizationStep.diagnostics_module ="
        " @SurfacePressureDiagnostics",
    ])
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
