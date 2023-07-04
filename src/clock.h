#ifndef CLOCK_H
#define CLOCK_H

#include <string>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <vector>


struct Clock {
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::unordered_map<std::string, std::chrono::duration<double>> total_times;
    std::unordered_map<std::string, int> total_calls;

    Clock() {}

    void tick(std::string name)
    {
        if (total_calls.find(name) == total_calls.end()) {
            total_calls[name] = 1;
            total_times[name] = std::chrono::duration<double>::zero();
        } else {
            total_calls[name]++;
        }

        start_times[name] = std::chrono::high_resolution_clock::now();
    }

    void tock(std::string name)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        total_times[name] += std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_times[name]);
    }

    void print()
    {
        Rcpp::Rcout << std::fixed;
        Rcpp::Rcout.precision(5);

        /*for (const auto& pair : total_times) {
         Rcpp::Rcout << pair.first << ' ' << pair.second.count() << " seconds\n";
        }*/

        std::vector<std::pair<std::string, std::chrono::duration<double>>> sorted_times(total_times.begin(), total_times.end());
        std::sort(sorted_times.begin(), sorted_times.end(), comparePairs);

        for (const auto& pair : sorted_times) {
            std::cout << pair.first << ": " << pair.second.count() << " seconds\n";
        }
    }

    static bool comparePairs(const std::pair<std::string, std::chrono::duration<double>>& lhs, const std::pair<std::string, std::chrono::duration<double>>& rhs)
    {
        return lhs.first < rhs.first;
    }

    void reset()
    {
        start_times.clear();
        total_times.clear();
        total_calls.clear();
    }
};


extern Clock CLOCK;

#endif
