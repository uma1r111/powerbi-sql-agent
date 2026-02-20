// frontend/src/components/Dashboard/ChartTypes/KPICard.jsx

import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const KPICard = ({ chart }) => {
    const { config, data } = chart;
    const value = config?.value || data?.[0]?.[Object.keys(data[0])[0]] || 0;
    const label = config?.label || chart.title || 'Metric';
    const trend = config?.trend;

    // Format value
    const formatValue = (val) => {
        if (typeof val === 'number') {
            // If it's a large number, format with commas
            if (val >= 1000000) {
                return `$${(val / 1000000).toFixed(2)}M`;
            } else if (val >= 1000) {
                return `$${val.toLocaleString()}`;
            } else {
                return val.toLocaleString();
            }
        }
        return val;
    };

    // Determine trend direction
    const getTrendIcon = () => {
        if (!trend) return null;
        if (trend > 0) return <TrendingUp className="w-5 h-5 text-green-500" />;
        if (trend < 0) return <TrendingDown className="w-5 h-5 text-red-500" />;
        return <Minus className="w-5 h-5 text-gray-400" />;
    };

    return (
        <div className="h-full flex flex-col justify-center items-center p-6 bg-gradient-to-br from-blue-50 to-white rounded-lg">
            <div className="text-sm font-medium text-gray-600 mb-2">{label}</div>
            <div className="text-4xl font-bold text-gray-900 mb-2">
                {formatValue(value)}
            </div>
            {trend !== undefined && (
                <div className="flex items-center gap-1">
                    {getTrendIcon()}
                    <span className={`text-sm font-medium ${trend > 0 ? 'text-green-600' : trend < 0 ? 'text-red-600' : 'text-gray-500'
                        }`}>
                        {trend > 0 ? '+' : ''}{trend}%
                    </span>
                </div>
            )}
        </div>
    );
};

export default KPICard;