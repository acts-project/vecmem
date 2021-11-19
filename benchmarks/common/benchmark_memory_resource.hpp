static void BenchmarkAllocation(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        std::string empty_string;
    }
}

static void BenchmarkDeallocation(benchmark::State& state) {
    for (auto _ : state) {
        std::string empty_string;
    }
}
