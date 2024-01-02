//
// Created by ianpo on 11/11/2023.
//

#pragma once

#include <cstdint>
#include <functional>

class UUID
{
public:
	UUID();
	UUID(uint64_t);
	UUID(const UUID&) = default;

	inline operator uint64_t() const { return m_UUID; }
private:
	uint64_t m_UUID;
};


namespace std
{
	template<>
	struct hash<UUID>
	{
		inline std::size_t operator()(const UUID& uuid) const
		{
			return hash<uint64_t>()(uuid);
		}
	};
} // namespace std

#define ML_RAND (static_cast<Real>(UUID()) / static_cast<Real>(std::numeric_limits<uint64_t>().max()))