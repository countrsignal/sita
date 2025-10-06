import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


###################################
# Classes
###################################

class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""
    
    def __init__(
        self,
        name: str,
        on_rank_zero: bool = False,
        extra: Optional[Mapping[str, object]] = None,
        ) -> None:
        
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param on_rank_zero: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)-8s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.on_rank_zero = on_rank_zero

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        # Set logging level if necessary
        if not self.isEnabledFor(level):
            self.setLevel(level=level)
        
        # process the message and kwargs
        msg, kwargs = self.process(msg, kwargs)
        
        # if DDP is being used, then rank_zero_only.rank will be set
        current_rank = getattr(rank_zero_only, "rank", None)
        if current_rank is not None:
            # > when the rank is set, then the rank will be prefixed to the message
            msg = rank_prefixed_message(msg, current_rank)
        
        # if on_rank_zero is True, then only rank zero will log
        if self.on_rank_zero:
            # > Only rank zero will log
            # NOTE: If rank is NOT set, then DDP is not being used
            #       - i.e., no multi-threading and therefore no rank
            #       - and the log will occur regardless
            if (current_rank == 0) or (current_rank is None):
                self.logger.log(level, msg, *args, **kwargs)
        else:
            # otherwise we allow other ranks to log
            if rank is None:
                # > All ranks will log
                self.logger.log(level, msg, *args, **kwargs)
            elif current_rank == rank:
                # > Only the specified rank will log
                self.logger.log(level, msg, *args, **kwargs)