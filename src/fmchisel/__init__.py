import logging

logger = logging.getLogger(__name__)

try:
    from llmcompressor.modifiers.factory import ModifierFactory

except ImportError:
    logger.info(
        "Optional dependency 'llmcompressor' not found; skipping FMCHISEL modifier "
        "registration. If you installed from source with the 'train' extra, this is expected."
    )

else:
    ModifierFactory._EXPERIMENTAL_PACKAGE_PATH = "fmchisel"
    ModifierFactory.refresh()
    logger.info("Registered FMCHISEL modifiers successfully.")
