// frontend/src/components/Dashboard/ChartTypes/DataTable.jsx

import React, { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

const DataTable = ({ chart, selectedKey, selectedValue, onSelect }) => {
    const { data, config } = chart;
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
    const [currentPage, setCurrentPage] = useState(1);

    const hasSelection = !!selectedValue;
    const isSelectable = !!onSelect;

    const rowsPerPage = 10;
    const columns = config?.columns || Object.keys(data[0] || {});
    const actualColumns = Object.keys(data[0] || {});

    // Sorting logic
    const sortedData = React.useMemo(() => {
        if (!sortConfig.key) return data;

        return [...data].sort((a, b) => {
            const aVal = a[sortConfig.key];
            const bVal = b[sortConfig.key];

            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
            }

            const aStr = String(aVal).toLowerCase();
            const bStr = String(bVal).toLowerCase();

            if (sortConfig.direction === 'asc') {
                return aStr.localeCompare(bStr);
            } else {
                return bStr.localeCompare(aStr);
            }
        });
    }, [data, sortConfig]);

    // Pagination
    const totalPages = Math.ceil(sortedData.length / rowsPerPage);
    const startIndex = (currentPage - 1) * rowsPerPage;
    const paginatedData = config?.pagination
        ? sortedData.slice(startIndex, startIndex + rowsPerPage)
        : sortedData;

    const handleSort = (column) => {
        setSortConfig(prev => ({
            key: column,
            direction: prev.key === column && prev.direction === 'asc' ? 'desc' : 'asc'
        }));
    };

    // Cross-filter: click a row to select it; click the same row again to clear
    const handleRowClick = (row) => {
        if (!isSelectable) return;
        const col = actualColumns[0];
        onSelect(col, row[col]);
    };

    const getRowClass = (row) => {
        if (!hasSelection || !selectedKey) return 'hover:bg-gray-50';
        const matchCol = actualColumns.find(c => c.toLowerCase() === selectedKey.toLowerCase());
        if (!matchCol) return 'hover:bg-gray-50';
        const isSelected = String(row[matchCol]) === selectedValue;
        if (isSelected) return 'bg-blue-50 border-l-2 border-blue-500 hover:bg-blue-100';
        return 'opacity-40 hover:opacity-70 hover:bg-gray-50';
    };

    const formatValue = (value) => {
        if (typeof value === 'number') {
            return value.toLocaleString();
        }
        return value;
    };

    return (
        <div className="h-full flex flex-col">
            <div className="flex-1 overflow-auto">
                <table className="w-full text-sm">
                    <thead className="bg-gray-50 sticky top-0">
                        <tr>
                            {actualColumns.map((col, idx) => (
                                <th
                                    key={col}
                                    className="px-4 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                    onClick={() => config?.sortable && handleSort(col)}
                                >
                                    <div className="flex items-center gap-2">
                                        <span>{columns[idx] || col}</span>
                                        {config?.sortable && sortConfig.key === col && (
                                            sortConfig.direction === 'asc'
                                                ? <ChevronUp className="w-4 h-4" />
                                                : <ChevronDown className="w-4 h-4" />
                                        )}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {paginatedData.map((row, rowIdx) => (
                            <tr
                                key={rowIdx}
                                className={`transition-all ${getRowClass(row)} ${isSelectable ? 'cursor-pointer' : ''}`}
                                onClick={() => handleRowClick(row)}
                            >
                                {actualColumns.map((col) => (
                                    <td key={col} className="px-4 py-3 whitespace-nowrap text-gray-900">
                                        {formatValue(row[col])}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Pagination */}
            {config?.pagination && totalPages > 1 && (
                <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200">
                    <div className="text-sm text-gray-700">
                        Showing {startIndex + 1} to {Math.min(startIndex + rowsPerPage, sortedData.length)} of {sortedData.length} results
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                            disabled={currentPage === 1}
                            className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Previous
                        </button>
                        <span className="px-3 py-1 text-sm">
                            Page {currentPage} of {totalPages}
                        </span>
                        <button
                            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                            disabled={currentPage === totalPages}
                            className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Next
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DataTable;