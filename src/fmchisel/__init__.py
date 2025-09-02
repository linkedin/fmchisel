import logging

logger = logging.getLogger(__name__)
try:
    from llmcompressor.modifiers.factory import ModifierFactory

    ModifierFactory._EXPERIMENTAL_PACKAGE_PATH = "fmchisel"
    ModifierFactory.refresh()
    logger.info("Registered FMCHISEL modifiers successfully.")
except ImportError:
    logger.info("llmcompressor not detected. FMCHISEL Modifiers will not be registered.")
