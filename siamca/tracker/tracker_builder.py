from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamca.core.config import cfg
from siamca.tracker.siamca_tracker import SiamCATracker

TRACKS = {
          'SiamCATracker': SiamCATracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
