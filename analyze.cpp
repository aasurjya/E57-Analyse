// #include <E57Format/E57SimpleReader.h>
// #include <iostream>
// #include <vector>
// #include <string>

// using namespace e57;
// using namespace std;

// int main(int argc, char** argv)
// {
//     if (argc < 2)
//     {
//         cerr << "Usage: ./analyze <file.e57>" << endl;
//         return 1;
//     }

//     string filename = argv[1];

//     try
//     {
//         Reader reader(filename);
//         E57Root header;
//         reader.GetE57Root(header);

//         cout << "========== E57 FILE ANALYSIS ==========\n";
//         cout << "File: " << filename << "\n";
//         cout << "Format Name: " << header.formatName << "\n";
//         cout << "Version: " << header.versionMajor << "." << header.versionMinor << "\n";
//         cout << "GUID: " << header.guid << "\n";
//         cout << "Scans: " << reader.GetData3DCount() << "\n";
//         cout << "Images: " << reader.GetImage2DCount() << "\n";
//         cout << "----------------------------------------\n";

//         // Iterate through each scan block
//         for (int64_t i = 0; i < reader.GetData3DCount(); i++)
//         {
//             Data3D data;
//             reader.ReadData3D(i, data);

//             cout << "\n--- Scan #" << i << " ---\n";
//             cout << "Name: " << data.name << "\n";
//             cout << "Description: " << data.description << "\n";
//             cout << "Point Count: " << data.pointCount << "\n";

//             cout << "Available Fields:\n";

//             if (data.pointFields.cartesianXField)
//                 cout << " - Cartesian X\n";
//             if (data.pointFields.cartesianYField)
//                 cout << " - Cartesian Y\n";
//             if (data.pointFields.cartesianZField)
//                 cout << " - Cartesian Z\n";

//             if (data.pointFields.sphericalRangeField)
//                 cout << " - Spherical Range\n";
//             if (data.pointFields.sphericalAzimuthField)
//                 cout << " - Spherical Azimuth\n";
//             if (data.pointFields.sphericalElevationField)
//                 cout << " - Spherical Elevation\n";

//             if (data.pointFields.intensityField)
//                 cout << " - Intensity\n";

//             if (data.pointFields.colorRedField && data.pointFields.colorGreenField && data.pointFields.colorBlueField)
//                 cout << " - RGB Color\n";

//             cout << "Pose (translation): "
//                  << data.pose.translation.x << ", "
//                  << data.pose.translation.y << ", "
//                  << data.pose.translation.z << "\n";

//             cout << "Pose (rotation quaternion): "
//                  << data.pose.rotation.w << ", "
//                  << data.pose.rotation.x << ", "
//                  << data.pose.rotation.y << ", "
//                  << data.pose.rotation.z << "\n";

//             // Optional: check if ready for Gaussian splats
//             bool okForGS = data.pointFields.cartesianXField && data.pointFields.cartesianYField && data.pointFields.cartesianZField;
//             cout << "Ready for Gaussian Splatting: " << (okForGS ? "YES" : "NO") << "\n";
//         }

//         cout << "\n========== END OF ANALYSIS ==========\n";
//     }
//     catch (E57Exception& e)
//     {
//         cerr << "\n========== E57 EXCEPTION ==========\n";
//         cerr << "Error: " << e.errorStr() << "\n";
//         cerr << "Error Code: " << e.errorCode() << "\n";
//         cerr << "Context: " << e.context() << "\n";
//         cerr << "Source: " << e.sourceFileName() << ":" << e.sourceLineNumber() << "\n";
//         cerr << "Function: " << e.sourceFunctionName() << "\n";
//         cerr << "===================================\n";
//         return 1;
//     }
//     catch (std::exception& e)
//     {
//         cerr << "Standard exception: " << e.what() << "\n";
//         return 1;
//     }

//     return 0;
// }


//Here

#include <E57Format/E57SimpleReader.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <limits>

using namespace e57;
using namespace std;

// Helper function to calculate bounding box from scan limits
struct BoundingBox {
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();
    
    double getVolume() const {
        double dx = maxX - minX;
        double dy = maxY - minY;
        double dz = maxZ - minZ;
        return (dx > 0 && dy > 0 && dz > 0) ? dx * dy * dz : 0.0;
    }
    
    double getDiagonal() const {
        double dx = maxX - minX;
        double dy = maxY - minY;
        double dz = maxZ - minZ;
        return sqrt(dx*dx + dy*dy + dz*dz);
    }
};

// Calculate estimated memory usage for Gaussian Splatting
double estimateMemoryUsage(int64_t pointCount) {
    // Rough estimate: each Gaussian needs ~100-200 bytes
    // (position, rotation quaternion, scale, opacity, SH coefficients)
    const double bytesPerGaussian = 150.0;
    return (pointCount * bytesPerGaussian) / (1024.0 * 1024.0); // MB
}

// Check if file exists and is readable
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Check if scan has organized/gridded structure
bool isOrganizedScan(const Data3D& data) {
    return (data.pointFields.rowIndexField || data.pointFields.columnIndexField);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cerr << "Usage: ./analyze <file.e57>" << endl;
        cerr << "\nThis tool analyzes E57 files for Gaussian Splatting conversion readiness." << endl;
        cerr << "\nExample: ./analyze 'BLK360 Outdoor Dataset.e57'" << endl;
        return 1;
    }

    string filename = argv[1];
    
    // Check if file exists
    if (!fileExists(filename)) {
        cerr << "Error: File '" << filename << "' not found or not readable." << endl;
        cerr << "Please check the file path and permissions." << endl;
        return 1;
    }

    try
    {
        Reader reader(filename);
        E57Root header;
        reader.GetE57Root(header);

        cout << "\n";
        cout << "╔══════════════════════════════════════════════════════════════╗\n";
        cout << "║         E57 FILE ANALYSIS FOR GAUSSIAN SPLATTING             ║\n";
        cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

        cout << "FILE INFORMATION:\n";
        cout << "  File: " << filename << "\n";
        cout << "  Format: " << header.formatName << "\n";
        cout << "  Version: " << header.versionMajor << "." << header.versionMinor << "\n";
        cout << "  GUID: " << header.guid << "\n";
        cout << "  Total Scans: " << reader.GetData3DCount() << "\n";
        cout << "  2D Images: " << reader.GetImage2DCount() << "\n";
        cout << string(64, '─') << "\n\n";

        // Overall readiness flags
        bool hasAnySuitableScan = false;
        bool hasAnyColorData = false;
        bool hasAnyImages = reader.GetImage2DCount() > 0;
        int64_t totalPoints = 0;
        BoundingBox overallBounds;
        double totalMemoryEstimate = 0.0;

        // Iterate through each scan block
        for (int64_t i = 0; i < reader.GetData3DCount(); i++)
        {
            Data3D data;
            reader.ReadData3D(i, data);

            cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
            cout << "║ SCAN #" << i << string(55 - to_string(i).length(), ' ') << "║\n";
            cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

            // Basic information
            cout << "BASIC INFO:\n";
            if (!data.name.empty())
                cout << "  Name: " << data.name << "\n";
            if (!data.description.empty())
                cout << "  Description: " << data.description << "\n";
            cout << "  Point Count: " << data.pointCount << "\n";
            totalPoints += data.pointCount;
            
            // Memory estimate for this scan
            double memoryMB = estimateMemoryUsage(data.pointCount);
            totalMemoryEstimate += memoryMB;
            cout << "  Est. Memory (GS): " << fixed << setprecision(1) << memoryMB << " MB\n";

            // Coordinate system
            cout << "\nCOORDINATE SYSTEM:\n";
            bool hasCartesian = data.pointFields.cartesianXField && 
                               data.pointFields.cartesianYField && 
                               data.pointFields.cartesianZField;
            bool hasSpherical = data.pointFields.sphericalRangeField && 
                               data.pointFields.sphericalAzimuthField && 
                               data.pointFields.sphericalElevationField;

            if (hasCartesian) {
                cout << "  ✓ Cartesian (X, Y, Z)\n";
                if (data.pointFields.cartesianInvalidStateField)
                    cout << "    - Has invalid state field\n";
            }
            if (hasSpherical) {
                cout << "  ✓ Spherical (Range, Azimuth, Elevation)\n";
                if (data.pointFields.sphericalInvalidStateField)
                    cout << "    - Has invalid state field\n";
            }
            if (!hasCartesian && !hasSpherical) {
                cout << "  ✗ No valid coordinate system detected!\n";
            }

            // Structure information
            cout << "\nSTRUCTURE:\n";
            bool isOrganized = isOrganizedScan(data);
            if (isOrganized) {
                cout << "  ✓ Organized/Gridded structure";
                if (data.pointFields.rowIndexField && data.pointFields.columnIndexField)
                    cout << " (Row/Column indexed)";
                cout << "\n";
            } else {
                cout << "  ○ Unorganized point cloud\n";
            }

            // Color information
            cout << "\nCOLOR DATA:\n";
            bool hasRGB = data.pointFields.colorRedField && 
                         data.pointFields.colorGreenField && 
                         data.pointFields.colorBlueField;
            bool hasIntensity = data.pointFields.intensityField;

            if (hasRGB) {
                cout << "  ✓ RGB Color (Full color information)\n";
                hasAnyColorData = true;
            }
            if (hasIntensity) {
                cout << "  ✓ Intensity values\n";
                if (!hasRGB) {
                    cout << "    ⚠ Note: Intensity can be used for grayscale visualization\n";
                    hasAnyColorData = true;
                }
            }
            if (!hasRGB && !hasIntensity) {
                cout << "  ✗ No color or intensity data\n";
            }

            // Additional attributes
            cout << "\nADDITIONAL ATTRIBUTES:\n";
            bool hasAnyExtras = false;

            if (data.pointFields.normalXField && data.pointFields.normalYField && data.pointFields.normalZField) {
                cout << "  ✓ Surface Normals (helpful but not required)\n";
                hasAnyExtras = true;
            }
            if (data.pointFields.timeStampField) {
                cout << "  ○ Time stamps\n";
                hasAnyExtras = true;
            }
            if (data.pointFields.returnIndexField || data.pointFields.returnCountField) {
                cout << "  ○ Multi-return data\n";
                hasAnyExtras = true;
            }
            if (!hasAnyExtras) {
                cout << "  - None detected\n";
            }

            // Pose information
            cout << "\nPOSE TRANSFORMATION:\n";
            cout << "  Translation: ("
                 << fixed << setprecision(4)
                 << data.pose.translation.x << ", "
                 << data.pose.translation.y << ", "
                 << data.pose.translation.z << ")\n";
            cout << "  Rotation (quaternion): ("
                 << data.pose.rotation.w << ", "
                 << data.pose.rotation.x << ", "
                 << data.pose.rotation.y << ", "
                 << data.pose.rotation.z << ")\n";
                 
            // Check if pose is identity (no transformation)
            bool isIdentityPose = (abs(data.pose.translation.x) < 1e-6 && 
                                 abs(data.pose.translation.y) < 1e-6 && 
                                 abs(data.pose.translation.z) < 1e-6 &&
                                 abs(data.pose.rotation.w - 1.0) < 1e-6 &&
                                 abs(data.pose.rotation.x) < 1e-6 &&
                                 abs(data.pose.rotation.y) < 1e-6 &&
                                 abs(data.pose.rotation.z) < 1e-6);
            if (isIdentityPose) {
                cout << "  Note: Identity pose (no transformation needed)\n";
            }

            // Gaussian Splatting readiness analysis
            cout << "\n" << string(64, '─') << "\n";
            cout << "GAUSSIAN SPLATTING READINESS:\n\n";

            bool gsReady = true;
            vector<string> requirements, recommendations, warnings;

            // Check essential requirements
            if (!hasCartesian && !hasSpherical) {
                gsReady = false;
                requirements.push_back("✗ CRITICAL: No valid 3D coordinates (need X,Y,Z or spherical)");
            } else if (hasSpherical && !hasCartesian) {
                recommendations.push_back("⚠ Spherical coordinates detected - conversion to Cartesian needed");
            }

            if (data.pointCount < 10000) {
                warnings.push_back("⚠ Low point count (<10k) - may result in sparse reconstruction");
            } else if (data.pointCount < 100000) {
                recommendations.push_back("○ Moderate point count - acceptable for small scenes");
            } else {
                recommendations.push_back("✓ Good point count for detailed reconstruction");
            }

            if (!hasRGB && !hasIntensity) {
                gsReady = false;
                requirements.push_back("✗ CRITICAL: No color/intensity data - Gaussian Splats require appearance info");
            } else if (hasIntensity && !hasRGB) {
                recommendations.push_back("⚠ Only intensity available - will produce grayscale splats");
            }

            if (isOrganized) {
                recommendations.push_back("✓ Organized structure - better for panorama reconstruction");
            }

            // Display analysis
            if (gsReady) {
                cout << "  STATUS: ✓ READY FOR CONVERSION\n\n";
                hasAnySuitableScan = true;
            } else {
                cout << "  STATUS: ✗ NOT READY - Missing critical data\n\n";
            }

            if (!requirements.empty()) {
                cout << "  Critical Requirements:\n";
                for (const auto& req : requirements)
                    cout << "    " << req << "\n";
                cout << "\n";
            }

            if (!recommendations.empty()) {
                cout << "  Recommendations:\n";
                for (const auto& rec : recommendations)
                    cout << "    " << rec << "\n";
                cout << "\n";
            }

            if (!warnings.empty()) {
                cout << "  Warnings:\n";
                for (const auto& warn : warnings)
                    cout << "    " << warn << "\n";
                cout << "\n";
            }

            cout << string(64, '─') << "\n";
        }

        // Overall summary
        cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
        cout << "║                     OVERALL SUMMARY                          ║\n";
        cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

        cout << "Total Points Across All Scans: " << totalPoints << "\n";
        cout << "Scans with 2D Images: " << reader.GetImage2DCount() << "\n";
        cout << "Estimated Total Memory (GS): " << fixed << setprecision(1) 
             << totalMemoryEstimate << " MB\n";
        if (totalMemoryEstimate > 1000) {
            cout << "  ⚠ Large memory footprint (>1GB) - consider downsampling\n";
        }
        cout << "\n";

        cout << "GAUSSIAN SPLATTING CONVERSION PATH:\n\n";

        if (hasAnySuitableScan) {
            cout << "✓ This E57 file can be converted to Gaussian Splatting format!\n\n";
            cout << "RECOMMENDED WORKFLOW:\n";
            cout << "  1. Export E57 → PLY format (with RGB colors)\n";
            cout << "     Tools: CloudCompare, E57Converter, Point Cloud Visualizer\n\n";
            cout << "  2. (Optional) Process point cloud:\n";
            cout << "     - Clean up noise and outliers\n";
            cout << "     - Estimate normals if missing\n";
            cout << "     - Subsample if density too high\n\n";
            cout << "  3. Train Gaussian Splatting model:\n";
            cout << "     Tools: 3D Gaussian Splatting (original), Nerfstudio, PostShot\n";
            if (hasAnyImages) {
                cout << "     Note: Embedded images can provide additional training data\n";
            }
            cout << "\n  4. Output formats:\n";
            cout << "     - Standard: .ply (with scale, rotation, opacity, SH coefficients)\n";
            cout << "     - Compressed: .splat, .spz (Niantic format)\n\n";
        } else {
            cout << "✗ This E57 file is NOT suitable for direct Gaussian Splatting conversion.\n\n";
            cout << "MISSING REQUIREMENTS:\n";
            if (!hasAnyColorData)
                cout << "  - No color or intensity data (essential for appearance)\n";
            cout << "\nCONSIDER:\n";
            cout << "  - Re-capturing scan with color/RGB camera\n";
            cout << "  - Using alternative 3D reconstruction methods (mesh-based)\n\n";
        }

        if (hasAnyImages && hasAnySuitableScan) {
            cout << "ALTERNATIVE APPROACH - Use Embedded Images:\n";
            cout << "  If this E57 contains panoramic images, you can:\n";
            cout << "  1. Extract panoramic/2D images from E57\n";
            cout << "  2. Use Structure-from-Motion (SfM) pipeline:\n";
            cout << "     - COLMAP for camera pose estimation\n";
            cout << "     - Direct Gaussian Splatting training from images\n";
            cout << "  This often produces BETTER results than point cloud initialization!\n\n";
        }

        cout << "KEY GAUSSIAN SPLATTING REQUIREMENTS:\n";
        cout << "  Essential:\n";
        cout << "    • 3D positions (X,Y,Z coordinates)\n";
        cout << "    • Color information (RGB) or intensity values\n";
        cout << "  Generated during training:\n";
        cout << "    • Scale (3D size of each Gaussian)\n";
        cout << "    • Rotation (quaternion orientation)\n";
        cout << "    • Opacity (transparency)\n";
        cout << "    • Spherical Harmonics (view-dependent color, optional)\n\n";

        cout << "For more information, visit:\n";
        cout << "  • https://github.com/graphdeco-inria/gaussian-splatting\n";
        cout << "  • https://radiancefields.com\n\n";

        cout << string(64, '═') << "\n";

    }
    catch (E57Exception& e)
    {
        cerr << "\n╔══════════════════════════════════════════════════════════════╗\n";
        cerr << "║                    E57 EXCEPTION ERROR                       ║\n";
        cerr << "╚══════════════════════════════════════════════════════════════╝\n\n";
        cerr << "Error: " << e.errorStr() << "\n";
        cerr << "Error Code: " << e.errorCode() << "\n";
        cerr << "Context: " << e.context() << "\n";
        cerr << "Source: " << e.sourceFileName() << ":" << e.sourceLineNumber() << "\n";
        cerr << "Function: " << e.sourceFunctionName() << "\n";
        cerr << string(64, '═') << "\n";
        return 1;
    }
    catch (std::exception& e)
    {
        cerr << "Standard exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}