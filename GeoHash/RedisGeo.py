from redis import StrictRedis


id_key = "pois"

conn = StrictRedis("localhost", 27017, db=4)

_id = conn.spop(id_key).decode()

while _id:
    results = conn.georadiusbymember(
        "geo_baidu", _id, 500, "m"
    )
    pipeline = conn.pipeline(transaction=False)

    conn.srem(id_key, *results)  # 第一种删除500米内的poi

    #  for 循环删除500米内点的key
    for b_id in results:
        pipeline.srem(id_key, b_id)
        pipeline.sadd("unique_ids", _id)
        pipeline.execute()
    try:
        _id = conn.spop(id_key).decode()
    except:
        break
