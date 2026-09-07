(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.Step4FailureTracker = api.Step4FailureTracker;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    class Step4FailureTracker {
        constructor(limit = 6) {
            this.limit = limit;
            this.resetWindow();
        }

        resetWindow() {
            this.consecutive = 0;
            this.tripped = false;
        }

        observe(result) {
            const isFailure = Boolean(result && result._networkError && !result._rateLimited);
            if (!isFailure) {
                this.consecutive = 0;
                return { consecutive: 0, tripped: this.tripped, tripNow: false };
            }

            if (!this.tripped) this.consecutive++;
            const tripNow = !this.tripped && this.consecutive >= this.limit;
            if (tripNow) this.tripped = true;
            return { consecutive: this.consecutive, tripped: this.tripped, tripNow };
        }
    }

    return { Step4FailureTracker };
}));
