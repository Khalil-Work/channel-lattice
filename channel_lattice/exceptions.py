"""Exception types for the channel_lattice package."""

from __future__ import annotations


class ChannelLatticeError(ValueError):
    """Raised for invalid channel-lattice configurations or usage.

    Every input-validation failure in :mod:`channel_lattice` raises this
    exception.  The message always states the violated constraint, the
    offending value, and the accepted range or convention, so that the
    caller can fix the input without consulting the documentation.
    """
