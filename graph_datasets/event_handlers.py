# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from pyscipopt import Eventhdlr
from pyscipopt import SCIP_EVENTTYPE

class DualBoundEventHandler(Eventhdlr):
    def __init__(self, initial_bound=None):
        super().__init__()
        self.initial_bound = initial_bound
        if initial_bound:
            self.events = [(initial_bound, 0, 0)]
            self.last_dual = initial_bound
        else:
            self.events = []
            self.last_dual = float("NaN")

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.LPEVENT, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.LPEVENT, self)

    def eventexec(self, event):
        dual = self.model.getDualbound()
        if dual != self.last_dual:
            if self.initial_bound:
                if self.model.getObjectiveSense() == "minimize":
                    dual = max(dual, self.initial_bound)
                else:
                    dual = min(dual, self.initial_bound)

            self.last_dual = dual
            time = self.model.getSolvingTime()
            nb_nodes = self.model.getNNodes()
            self.events.append((dual, time, nb_nodes))
            #print(f"CAUGHT EVENT {dual} at t={time} nb_nodes={nb_nodes}", flush=True)
        return {}