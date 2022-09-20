/******************************************************************************
**                                                                           **
** Copyright (C) Joyson Electronics (2022)                                   **
**                                                                           **
** All rights reserved.                                                      **
**                                                                           **
** This document contains proprietary information belonging to Joyson        **
** Electronics. Passing on and copying of this document, and communication   **
** of its contents is not permitted without prior written authorization.     **
**                                                                           **
******************************************************************************/

#pragma once

#include <iostream>
#include <vector>


namespace hdmap {

struct Point
{
    double x = 0;
    double y = 0;
};

class Ploygon
{   
    public:
    std::vector<Point> pt;
    // unsigned EdgeNumber();

    unsigned EdgeNumber() 
    {
        return this->pt.size();
    }
};

struct DrivableArea
{
    std::vector<Ploygon> pts;
};

struct RoadSegment
{
    std::vector<Ploygon> pts;
    bool is_intersection;
    DrivableArea drivable_area;
};

struct LaneDividerSegments
{
    Point pt;
    int8_t segment_type;
};

struct LaneInfo
{
    std::vector<Ploygon> pts;
    int8_t lane_type;
    int from_edge_line;
    int to_edge_line;
    LaneDividerSegments left_lane_divider_segments;
    LaneDividerSegments right_lane_divider_segments;
};

struct CrosswalkInfo
{
    std::vector<Ploygon> pts;
    RoadSegment road_segment;
};

struct StopLineInfo
{
    std::vector<Ploygon> pts;
    int8_t stop_line_type;
    std::vector<CrosswalkInfo> crosswalks;
};

struct LaneConnector
{
    std::vector<Ploygon> pts;
};

} // namespace hdmap