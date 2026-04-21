// frontend/src/components/Dashboard/FilterPanel.jsx

import React from 'react';
import { X, Filter } from 'lucide-react';

const FilterPanel = ({ filters, onRemoveFilter, onClearAll }) => {
    const filterEntries = Object.entries(filters || {});

    if (filterEntries.length === 0) {
        return null;
    }

    return (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-900">Active Filters</span>
                </div>
                <button
                    onClick={onClearAll}
                    className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                >
                    Clear All
                </button>
            </div>

            <div className="flex flex-wrap gap-2">
                {filterEntries.map(([key, value]) => (
                    <div
                        key={key}
                        className="flex items-center gap-2 bg-white px-3 py-1 rounded-full border border-blue-300 text-sm"
                    >
                        <span className="text-gray-700">
                            <span className="font-medium">{key}:</span> {value}
                        </span>
                        <button
                            onClick={() => onRemoveFilter(key)}
                            className="hover:bg-blue-100 rounded-full p-0.5 transition-colors"
                        >
                            <X className="w-3 h-3 text-gray-600" />
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default FilterPanel;