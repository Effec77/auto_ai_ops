/**
 * AidFlow AI - Frontend Integration Guide
 * Enhanced route visualization with risk-based styling and tooltips
 */

// ========== ROUTE STYLING FUNCTIONS ==========

/**
 * Dynamic styling function for routes based on risk level and type
 * @param {Object} feature - GeoJSON feature with properties
 * @returns {Object} Leaflet style options
 */
function styleRoute(feature) {
    const props = feature.properties;

    // Use color from backend if available, otherwise calculate
    const color = props.color || calculateRouteColor(props.risk_level, props.rerouted);

    return {
        color: color,
        weight: props.weight || (props.rerouted ? 5 : 4),
        opacity: props.opacity || 0.8,
        dashArray: props.dashArray || (props.rerouted ? "10,5" : null),
        lineCap: 'round',
        lineJoin: 'round'
    };
}

/**
 * Calculate route color based on risk level and reroute status
 * @param {number} riskLevel - Risk score (0-150+)
 * @param {boolean} isRerouted - Whether route is rerouted
 * @returns {string} Hex color code
 */
function calculateRouteColor(riskLevel, isRerouted) {
    if (isRerouted) {
        if (riskLevel <= 25) return "#fbbf24";  // Yellow - low risk reroute
        if (riskLevel <= 75) return "#f97316";  // Orange - medium risk reroute
        return "#dc2626";                       // Red - high risk reroute
    } else {
        if (riskLevel === 0) return "#2563eb";  // Blue - safe primary route
        if (riskLevel <= 25) return "#16a34a";  // Green - low risk primary
        if (riskLevel <= 75) return "#f59e0b";  // Amber - medium risk primary
        return "#dc2626";                       // Red - high risk primary
    }
}

// ========== TOOLTIP FUNCTIONS ==========

/**
 * Create interactive tooltips with risk information
 * @param {Object} feature - GeoJSON feature
 * @param {Object} layer - Leaflet layer
 */
function onEachRoute(feature, layer) {
    const props = feature.properties;

    // Create tooltip content
    const tooltipContent = createRouteTooltip(props);

    // Bind tooltip with hover behavior
    layer.bindTooltip(tooltipContent, {
        permanent: false,
        direction: 'top',
        offset: [0, -10],
        className: 'route-tooltip'
    });

    // Add click popup with detailed information
    const popupContent = createRoutePopup(props);
    layer.bindPopup(popupContent, {
        maxWidth: 300,
        className: 'route-popup'
    });

    // Add hover effects
    layer.on({
        mouseover: function (e) {
            this.setStyle({
                weight: (props.weight || 4) + 2,
                opacity: 1.0
            });
        },
        mouseout: function (e) {
            this.setStyle({
                weight: props.weight || 4,
                opacity: props.opacity || 0.8
            });
        }
    });
}

/**
 * Create tooltip content for route hover
 * @param {Object} props - Route properties
 * @returns {string} HTML tooltip content
 */
function createRouteTooltip(props) {
    const routeTypeIcon = getRouteTypeIcon(props.route_type, props.rerouted);
    const riskIcon = props.risk_level >= 15 ? "🚨" : "✅";

    return `
        <div class="route-tooltip-content">
            <strong>${routeTypeIcon} ${props.route_type.charAt(0).toUpperCase() + props.route_type.slice(1)} Route</strong><br>
            ${riskIcon} Risk Level: ${props.risk_level}<br>
            📏 Distance: ${props.distance_km.toFixed(1)} km
        </div>
    `;
}

/**
 * Create detailed popup content for route click
 * @param {Object} props - Route properties
 * @returns {string} HTML popup content
 */
function createRoutePopup(props) {
    const routeTypeIcon = getRouteTypeIcon(props.route_type, props.rerouted);
    const safetyStatus = props.is_safe ? "✅ Safe Route" : "🚨 Compromised Route";
    const riskLevel = getRiskLevelText(props.risk_level);

    let content = `
        <div class="route-popup-content">
            <h3>${routeTypeIcon} ${props.name}</h3>
            <div class="route-stats">
                <p><strong>Status:</strong> ${safetyStatus}</p>
                <p><strong>Distance:</strong> ${props.distance_km.toFixed(1)} km</p>
                <p><strong>Duration:</strong> ${props.duration_hours.toFixed(1)} hours</p>
                <p><strong>Risk Level:</strong> ${riskLevel} (${props.risk_level})</p>
            </div>
    `;

    // Add disaster information if present
    if (props.disaster_scenario) {
        content += `
            <div class="disaster-info">
                <p><strong>Disaster:</strong> ${props.disaster_scenario}</p>
                <p><strong>Affected Zones:</strong> ${props.affected_zones}</p>
            </div>
        `;
    }

    // Add reroute information
    if (props.rerouted) {
        const rerouteStatus = props.reroute_success ? "✅ Successfully Rerouted" : "❌ Reroute Failed";
        content += `
            <div class="reroute-info">
                <p><strong>Reroute Status:</strong> ${rerouteStatus}</p>
                ${props.original_risk ? `<p><strong>Original Risk:</strong> ${props.original_risk}</p>` : ''}
            </div>
        `;
    }

    content += `</div>`;
    return content;
}

/**
 * Get icon for route type
 * @param {string} routeType - Route type (original, rerouted, compromised)
 * @param {boolean} isRerouted - Whether route is rerouted
 * @returns {string} Emoji icon
 */
function getRouteTypeIcon(routeType, isRerouted) {
    switch (routeType) {
        case 'original': return '📍';
        case 'rerouted': return '🔄';
        case 'compromised': return '⚠️';
        default: return isRerouted ? '🔄' : '📍';
    }
}

/**
 * Get human-readable risk level text
 * @param {number} riskLevel - Numeric risk level
 * @returns {string} Risk level description
 */
function getRiskLevelText(riskLevel) {
    if (riskLevel === 0) return "No Risk";
    if (riskLevel <= 25) return "Low Risk";
    if (riskLevel <= 75) return "Medium Risk";
    if (riskLevel <= 150) return "High Risk";
    return "Critical Risk";
}

// ========== MAP INTEGRATION ==========

/**
 * Load and display routes on the map
 * @param {Object} map - Leaflet map instance
 * @param {string} geojsonUrl - URL to GeoJSON file
 * @param {string} layerName - Name for the layer
 */
function loadRouteLayer(map, geojsonUrl, layerName) {
    fetch(geojsonUrl)
        .then(response => response.json())
        .then(data => {
            const routeLayer = L.geoJSON(data, {
                style: styleRoute,
                onEachFeature: onEachRoute
            }).addTo(map);

            // Add to layer control if it exists
            if (window.layerControl) {
                window.layerControl.addOverlay(routeLayer, layerName);
            }

            // Fit map to route bounds
            if (data.features.length > 0) {
                map.fitBounds(routeLayer.getBounds(), { padding: [20, 20] });
            }
        })
        .catch(error => {
            console.error('Error loading route:', error);
        });
}

/**
 * Load multiple route files for comparison
 * @param {Object} map - Leaflet map instance
 * @param {Array} routeFiles - Array of {url, name, type} objects
 */
function loadMultipleRoutes(map, routeFiles) {
    const routeLayers = {};

    routeFiles.forEach(routeFile => {
        fetch(routeFile.url)
            .then(response => response.json())
            .then(data => {
                const layer = L.geoJSON(data, {
                    style: styleRoute,
                    onEachFeature: onEachRoute
                });

                routeLayers[routeFile.name] = layer;

                // Add to map based on route type
                if (routeFile.type === 'original') {
                    layer.addTo(map);
                }
            });
    });

    return routeLayers;
}

// ========== CSS STYLES ==========

const routeStyles = `
<style>
.route-tooltip {
    background: rgba(0, 0, 0, 0.8);
    border: none;
    border-radius: 4px;
    color: white;
    font-size: 12px;
    padding: 8px;
}

.route-popup {
    font-family: Arial, sans-serif;
}

.route-popup-content h3 {
    margin: 0 0 10px 0;
    color: #333;
    font-size: 16px;
}

.route-stats, .disaster-info, .reroute-info {
    margin: 10px 0;
    padding: 8px;
    background: #f8f9fa;
    border-radius: 4px;
}

.route-stats p, .disaster-info p, .reroute-info p {
    margin: 4px 0;
    font-size: 14px;
}

.disaster-info {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
}

.reroute-info {
    background: #dcfce7;
    border-left: 4px solid #16a34a;
}
</style>
`;

// ========== USAGE EXAMPLE ==========

/**
 * Example usage for AidFlow AI dashboard
 */
function initializeAidFlowRoutes(map) {
    // Add CSS styles
    document.head.insertAdjacentHTML('beforeend', routeStyles);

    // Load route files
    const routeFiles = [
        {
            url: 'route_exports/himachal_to_maharashtra_alt_1.geojson',
            name: 'Original Route',
            type: 'original'
        },
        {
            url: 'route_exports/himachal_to_maharashtra_rerouted_major_flooding.geojson',
            name: 'Rerouted (Flooding)',
            type: 'rerouted'
        },
        {
            url: 'route_exports/himachal_to_maharashtra_compromised_bridge_collapse.geojson',
            name: 'Compromised (Bridge)',
            type: 'compromised'
        }
    ];

    // Load and display routes
    const routeLayers = loadMultipleRoutes(map, routeFiles);

    // Create layer control for toggling routes
    const overlayMaps = {};
    Object.keys(routeLayers).forEach(name => {
        overlayMaps[name] = routeLayers[name];
    });

    L.control.layers(null, overlayMaps, {
        position: 'topright',
        collapsed: false
    }).addTo(map);
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        styleRoute,
        onEachRoute,
        loadRouteLayer,
        loadMultipleRoutes,
        initializeAidFlowRoutes
    };
}