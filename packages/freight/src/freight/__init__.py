"""The freight: reads a request, chooses in a depot's snapshot (kinds,
shares, the depot's order), runs one preparation on what it chose, leaves
the result on the dock with a delivery note, finds a delivery again, and
marks what an erasure reached (docs/api/freight.md).

It reads the depot only through the depot's api module, and knows nothing
of what its preparations make: a preparation is a program it starts and
talks to (docs/api/preparation.md). Code outside the freight imports only
freight.api.

    api          what code outside the freight asks of it
    selection    what a request chooses, the census and the delivery key
    fill         how a part fills its budget an item at a time
    groups       how a share splits its budget over the tags of its texts
    job          one request run, from the snapshot to the delivery note
    exchange     the program of a preparation, started and talked to
    messages     the messages of that exchange on byte streams
    deliveries   the deliveries on the dock: notes, staleness, the cap
    erasures     what the erasures a depot carried out did to the deliveries
    jobs         the queue, one job at a time
"""
