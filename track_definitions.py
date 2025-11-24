"""
Track corner definitions for all circuits
"""

BARBER_CORNERS = {
    'T1': {
        'name': 'Turn 1',
        'start': 0,
        'end': 150,
        'apex': 75,
        'type': 'slow',
        'direction': 'Right',
    },
    'T2': {
        'name': 'Turn 2',
        'start': 400,
        'end': 550,
        'apex': 475,
        'type': 'medium',
        'direction': 'Left',
    },
    'T3': {
        'name': 'Turn 3',
        'start': 700,
        'end': 850,
        'apex': 775,
        'type': 'medium',
        'direction': 'Right',
    },
    'T4': {
        'name': 'Turn 4',
        'start': 1000,
        'end': 1150,
        'apex': 1075,
        'type': 'fast',
        'direction': 'Left',
    },
    'T5': {
        'name': 'Turn 5',
        'start': 1300,
        'end': 1500,
        'apex': 1400,
        'type': 'medium',
        'direction': 'Right',
    },
    'T6': {
        'name': 'Turn 6',
        'start': 1600,
        'end': 1750,
        'apex': 1675,
        'type': 'slow',
        'direction': 'Left',
    },
    'T7': {
        'name': 'Turn 7',
        'start': 1900,
        'end': 2050,
        'apex': 1975,
        'type': 'slow',
        'direction': 'Right',
    },
    'T8': {
        'name': 'Turn 8',
        'start': 2200,
        'end': 2350,
        'apex': 2275,
        'type': 'medium',
        'direction': 'Left',
    },
    'T9': {
        'name': 'Turn 9',
        'start': 2500,
        'end': 2650,
        'apex': 2575,
        'type': 'medium',
        'direction': 'Right',
    },
    'T10': {
        'name': 'Turn 10',
        'start': 2800,
        'end': 2950,
        'apex': 2875,
        'type': 'fast',
        'direction': 'Left',
    },
    'T11': {
        'name': 'Turn 11',
        'start': 3050,
        'end': 3200,
        'apex': 3125,
        'type': 'fast',
        'direction': 'Right',
    },
    'T12': {
        'name': 'Turn 12',
        'start': 3300,
        'end': 3400,
        'apex': 3350,
        'type': 'medium',
        'direction': 'Left',
    },
    'T13': {
        'name': 'Turn 13',
        'start': 3450,
        'end': 3600,
        'apex': 3525,
        'type': 'slow',
        'direction': 'Right',
    },
}

INDIANAPOLIS_CORNERS = {
    'T1': {
        'name': 'Turn 1',
        'start': 496,
        'end': 693,
        'apex': 672,
        'type': 'slow',
        'direction': 'Right',
    },
    'T2': {
        'name': 'Turn 2',
        'start': 700,
        'end': 850,
        'apex': 750,
        'type': 'medium',
        'direction': 'Left',
    },
    'T3': {
        'name': 'Turn 3',
        'start': 900,
        'end': 1100,
        'apex': 1000,
        'type': 'fast',
        'direction': 'Right',
    },
    'T4': {
        'name': 'Turn 4',
        'start': 1291,
        'end': 1363,
        'apex': 1331,
        'type': 'medium',
        'direction': 'Right',
    },
    'T5': {
        'name': 'Turn 5',
        'start': 1697,
        'end': 1895,
        'apex': 1887,
        'type': 'fast',
        'direction': 'Left',
    },
    'T6': {
        'name': 'Turn 6',
        'start': 1905,
        'end': 2050,
        'apex': 1950,
        'type': 'slow',
        'direction': 'Right',
    },
    'T7': {
        'name': 'Turn 7',
        'start': 2060,
        'end': 2180,
        'apex': 2114,
        'type': 'slow',
        'direction': 'Left',
    },
    'T8': {
        'name': 'Turn 8',
        'start': 2171,
        'end': 2269,
        'apex': 2186,
        'type': 'medium',
        'direction': 'Right',
    },
    'T9': {
        'name': 'Turn 9',
        'start': 2283,
        'end': 2364,
        'apex': 2318,
        'type': 'medium',
        'direction': 'Left',
    },
    'T10': {
        'name': 'Turn 10',
        'start': 2470,
        'end': 2510,
        'apex': 2470,
        'type': 'medium',
        'direction': 'Right',
    },
    'T11': {
        'name': 'Turn 11',
        'start': 2703,
        'end': 2746,
        'apex': 2746,
        'type': 'fast',
        'direction': 'Right',
    },
    'T12': {
        'name': 'Turn 12',
        'start': 2764,
        'end': 2814,
        'apex': 2779,
        'type': 'fast',
        'direction': 'Right',
    },
    'T13': {
        'name': 'Turn 13',
        'start': 2992,
        'end': 3176,
        'apex': 3087,
        'type': 'slow',
        'direction': 'Right',
    },
    'T14': {
        'name': 'Turn 14',
        'start': 3158,
        'end': 3340,
        'apex': 3238,
        'type': 'medium',
        'direction': 'Left',
    },
    'T15': {
        'name': 'Turn 15',
        'start': 3345,
        'end': 3696,
        'apex': 3412,
        'type': 'medium',
        'direction': 'Right',
    },
}


def get_track_corners(track_name):
    """Get corner definitions for a track"""
    tracks = {
        'barber': BARBER_CORNERS,
        'indianapolis': INDIANAPOLIS_CORNERS,
        # Add more tracks as needed
    }
    
    if track_name.lower() not in tracks:
        # Return default (Barber) if track not found
        return BARBER_CORNERS
    
    return tracks[track_name.lower()]
