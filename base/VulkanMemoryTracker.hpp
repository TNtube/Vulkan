#pragma once

#include <vulkan/vulkan.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace vks
{

struct AllocationInfo {
	VkDeviceSize size;
	uint32_t memoryTypeIndex;
	std::string tag;
};

class MemoryTracker
{
private:
	std::unordered_map<VkDeviceMemory, AllocationInfo> allocations;
	std::mutex mutex;
	VkDeviceSize totalAllocated = 0;
	VkDeviceSize peakAllocated = 0;

	// Per-tag tracking
	std::unordered_map<std::string, VkDeviceSize> taggedAllocations;

	// Singleton
	MemoryTracker() = default;

public:
	static MemoryTracker& getInstance() {
		static MemoryTracker instance;
		return instance;
	}

	MemoryTracker(const MemoryTracker&) = delete;
	MemoryTracker& operator=(const MemoryTracker&) = delete;

	void recordAllocation(VkDeviceMemory memory, VkDeviceSize size, uint32_t memoryTypeIndex, const std::string& tag = "") {
		std::lock_guard<std::mutex> lock(mutex);

		AllocationInfo info{ size, memoryTypeIndex, tag };
		allocations[memory] = info;

		totalAllocated += size;
		if (totalAllocated > peakAllocated) {
			peakAllocated = totalAllocated;
		}

		if (!tag.empty()) {
			taggedAllocations[tag] += size;
		}
	}

	void recordFree(VkDeviceMemory memory) {
		std::lock_guard<std::mutex> lock(mutex);

		auto it = allocations.find(memory);
		if (it != allocations.end()) {
			totalAllocated -= it->second.size;
			if (!it->second.tag.empty()) {
				taggedAllocations[it->second.tag] -= it->second.size;
			}
			allocations.erase(it);
		}
	}

	VkDeviceSize getTotalAllocated() const {
		return totalAllocated;
	}

	VkDeviceSize getPeakAllocated() const {
		return peakAllocated;
	}

	VkDeviceSize getAllocationByTag(const std::string& tag) const {
		auto it = taggedAllocations.find(tag);
		return (it != taggedAllocations.end()) ? it->second : 0;
	}

	size_t getAllocationCount() const {
		return allocations.size();
	}

	void reset() {
		std::lock_guard<std::mutex> lock(mutex);
		allocations.clear();
		taggedAllocations.clear();
		totalAllocated = 0;
		peakAllocated = 0;
	}

	void printSummary() const {
		std::cout << std::fixed << std::setprecision(2);
		std::cout << "\n=== GPU Memory Summary ===" << std::endl;
		std::cout << "Total allocated: " << (totalAllocated / (1024.0 * 1024.0)) << " MB" << std::endl;
		std::cout << "Peak allocated:  " << (peakAllocated / (1024.0 * 1024.0)) << " MB" << std::endl;
		std::cout << "Allocation count: " << allocations.size() << std::endl;

		if (!taggedAllocations.empty()) {
			std::cout << "\nBy tag:" << std::endl;
			for (const auto& [tag, size] : taggedAllocations) {
				if (size > 0) {
					std::cout << "  " << tag << ": " << (size / (1024.0 * 1024.0)) << " MB" << std::endl;
				}
			}
		}
		std::cout << "==========================\n" << std::endl;
	}

	void saveToCSV(const std::string& filename) const {
		std::ofstream file(filename);
		if (!file.is_open()) return;

		file << "metric,value_bytes,value_mb" << std::endl;
		file << "total_allocated," << totalAllocated << "," << (totalAllocated / (1024.0 * 1024.0)) << std::endl;
		file << "peak_allocated," << peakAllocated << "," << (peakAllocated / (1024.0 * 1024.0)) << std::endl;
		file << "allocation_count," << allocations.size() << "," << allocations.size() << std::endl;

		if (!taggedAllocations.empty()) {
			file << std::endl << "tag,size_bytes,size_mb" << std::endl;
			for (const auto& [tag, size] : taggedAllocations) {
				file << tag << "," << size << "," << (size / (1024.0 * 1024.0)) << std::endl;
			}
		}

		file.close();
	}
};

// Convenience macros for tracking allocations with tags
#define VKS_TRACK_ALLOC(memory, size, memTypeIndex, tag) \
	vks::MemoryTracker::getInstance().recordAllocation(memory, size, memTypeIndex, tag)

#define VKS_TRACK_FREE(memory) \
	vks::MemoryTracker::getInstance().recordFree(memory)

#define VKS_MEMORY_SUMMARY() \
	vks::MemoryTracker::getInstance().printSummary()

#define VKS_MEMORY_SAVE_CSV(filename) \
	vks::MemoryTracker::getInstance().saveToCSV(filename)

} // namespace vks