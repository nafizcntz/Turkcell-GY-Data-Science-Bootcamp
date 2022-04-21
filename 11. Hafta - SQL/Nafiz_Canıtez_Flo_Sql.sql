set language Turkish

CREATE DATABASE CUSTOMERS;

----------------------------------------------------
-- T�P D�N���MLER�
----------------------------------------------------

------------- Date Kolonlar� Tip D�n���m� -------------
ALTER TABLE FLO
ADD first_order_date1 DATE;
UPDATE FLO
SET first_order_date1 = CAST(first_order_date AS DATE) FROM FLO
ALTER TABLE FLO
DROP COLUMN first_order_date;

ALTER TABLE FLO
ADD last_order_date1 DATE;
UPDATE FLO
SET last_order_date1 = CAST(last_order_date AS DATE) FROM FLO
ALTER TABLE FLO
DROP COLUMN last_order_date;

ALTER TABLE FLO
ADD last_order_date_online1 DATE;
UPDATE FLO
SET last_order_date_online1 = CAST(last_order_date_online AS DATE) FROM FLO
ALTER TABLE FLO
DROP COLUMN last_order_date_online;

ALTER TABLE FLO
ADD last_order_date_offline1 DATE;
UPDATE FLO
SET last_order_date_offline1 = CAST(last_order_date_offline AS DATE) FROM FLO
ALTER TABLE FLO
DROP COLUMN last_order_date_offline;

------------- �nt Kolonlar� Tip D�n���m� -------------

ALTER TABLE FLO
ALTER COLUMN order_num_total_ever_online int;

ALTER TABLE FLO
ALTER COLUMN order_num_total_ever_offline int;

------------- Float Kolonlar� Tip D�n���m� -------------

ALTER TABLE FLO
ALTER COLUMN customer_value_total_ever_offline float;

ALTER TABLE FLO
ALTER COLUMN customer_value_total_ever_online float;

----------------------------------------------------
-- �DEV SORULARI
----------------------------------------------------

-- 2. Soru --
SELECT SUM(customer_value_total_ever_online) FROM FLO
SELECT SUM(customer_value_total_ever_offline) FROM FLO
SELECT SUM(customer_value_total_ever_online + customer_value_total_ever_offline) FROM FLO

-- 3. Soru --
SELECT master_id, (SUM(customer_value_total_ever_online + customer_value_total_ever_offline)/SUM(order_num_total_ever_online + order_num_total_ever_offline)) Mean FROM FLO
GROUP BY master_id

-- 4. Soru --
SELECT last_order_channel, SUM(customer_value_total_ever_online + customer_value_total_ever_offline) Order_Channels_Sum FROM FLO 
GROUP BY last_order_channel

-- 5. Soru --
SELECT SUM(order_num_total_ever_online + order_num_total_ever_offline) Toplam_Fatura_Adedi FROM FLO

-- 6. Soru --
SELECT last_order_channel, SUM(order_num_total_ever_online + order_num_total_ever_offline) FROM FLO 
GROUP BY last_order_channel

-- 7. Soru --
SELECT SUM(order_num_total_ever_online + order_num_total_ever_offline) Toplam_�r�n_Say�s� FROM FLO

-- 8. Soru --
SELECT DATEPART(YEAR, last_order_date1) Y�l, SUM(order_num_total_ever_online + order_num_total_ever_offline) Toplam_�r�n_Adedi FROM FLO
GROUP BY DATEPART(YEAR, last_order_date1)

SELECT DATEPART(YEAR, first_order_date1) Y�l, SUM(order_num_total_ever_online + order_num_total_ever_offline) Toplam_�r�n_Adedi FROM FLO
GROUP BY DATEPART(YEAR, first_order_date1)

-- 9. Soru --
SELECT order_channel Patform, AVG(order_num_total_ever_online + order_num_total_ever_offline) Ort_�r�n_Adedi FROM FLO
GROUP BY order_channel

-- 10. Soru --
SELECT COUNT(DISTINCT master_id) Ka�_Farkl�_Ki�i FROM FLO

-- 11. Soru --
SELECT TOP 1 
interested_in_categories_12, COUNT(interested_in_categories_12) En_�ok_�lgi FROM FLO
GROUP BY interested_in_categories_12
ORDER BY En_�ok_�lgi DESC

-- 12. Soru --
SELECT order_channel, interested_in_categories_12, COUNT(interested_in_categories_12) En_�ok_�lgi FROM FLO
GROUP BY order_channel, interested_in_categories_12
ORDER BY order_channel, interested_in_categories_12 DESC

-- 13. Soru --
SELECT store_type, COUNT(store_type) En_�ok_Tercih_Edilen FROM FLO
GROUP BY store_type
ORDER BY En_�ok_Tercih_Edilen DESC

-- 14. Soru --
SELECT store_type, SUM(customer_value_total_ever_online + customer_value_total_ever_offline) Toplam_Ciro FROM FLO
GROUP BY store_type
ORDER BY Toplam_Ciro DESC

-- 15. Soru --
SELECT TOP 4
order_channel, store_type, COUNT(store_type) En_�ok_Tercih_Edilen FROM FLO
GROUP BY order_channel, store_type
ORDER BY En_�ok_Tercih_Edilen DESC

-- 16. Soru --
SELECT TOP 1 
master_id, SUM(order_num_total_ever_online + order_num_total_ever_offline) En_�ok_Al��veri� FROM FLO
GROUP BY master_id
ORDER BY En_�ok_Al��veri� DESC

-- 17. Soru --
SELECT TOP 1 
master_id, (SUM(customer_value_total_ever_online + customer_value_total_ever_offline)/SUM(order_num_total_ever_online + order_num_total_ever_offline)) En_�ok_Al��veri� FROM FLO
GROUP BY master_id
ORDER BY SUM(order_num_total_ever_online + order_num_total_ever_offline) DESC

-- 18. Soru --
SELECT TOP 1
master_id, (SUM(order_num_total_ever_online + order_num_total_ever_offline)/(DATEDIFF(DAY, first_order_date1, last_order_date1)+0.01)) AS DateDif FROM FLO
GROUP BY master_id, DATEDIFF(DAY, first_order_date1, last_order_date1)
ORDER BY SUM(order_num_total_ever_online + order_num_total_ever_offline) DESC

-- 19. Soru --
SELECT TOP 100 
master_id, 
SUM(order_num_total_ever_online + order_num_total_ever_offline) AS En_�ok_Al��veri�,
(SUM(order_num_total_ever_online + order_num_total_ever_offline)/(DATEDIFF(DAY, first_order_date1, last_order_date1)+0.01)) AS DateDif 
FROM FLO
GROUP BY master_id, (DATEDIFF(DAY, first_order_date1, last_order_date1))
ORDER BY SUM(customer_value_total_ever_offline+customer_value_total_ever_online) DESC

-- 20. Soru --
SELECT TOP 1 
master_id, order_channel, SUM(order_num_total_ever_online + order_num_total_ever_offline) En_�ok_Al��veri� FROM FLO
GROUP BY master_id, order_channel
ORDER BY En_�ok_Al��veri� DESC

-- 21. Soru --
SELECT master_id, last_order_date1 FROM FLO a
WHERE a.last_order_date1 = (SELECT MAX(last_order_date1) FROM FLO)

-- 22. Soru --
SELECT master_id, last_order_date1, (SUM(order_num_total_ever_online + order_num_total_ever_offline)/(DATEDIFF(DAY, first_order_date1, last_order_date1)+0.01)) AS DateDif 
FROM FLO flo
WHERE flo.last_order_date1 = (SELECT MAX(last_order_date1) FROM FLO)
GROUP BY master_id, last_order_date1, DATEDIFF(DAY, first_order_date1, last_order_date1)
ORDER BY DateDif, DATEDIFF(DAY, first_order_date1, last_order_date1) DESC

-- 23. Soru --
SELECT TOP 4
master_id, order_channel, (SUM(customer_value_total_ever_online + customer_value_total_ever_offline)/SUM(order_num_total_ever_online + order_num_total_ever_offline)) AS DateDif 
FROM FLO flo
WHERE flo.last_order_date1 = (SELECT MAX(last_order_date1) FROM FLO)
GROUP BY master_id, order_channel
ORDER BY DateDif DESC

-- 24. Soru --
SELECT master_id, first_order_date1 FROM FLO a
WHERE a.first_order_date1 = (SELECT MIN(first_order_date1) FROM FLO)

-- 25. Soru --
SELECT master_id, first_order_date1, (SUM(order_num_total_ever_online + order_num_total_ever_online)/(DATEDIFF(day, first_order_date1, last_order_date1))) AS DateDif FROM FLO flo
WHERE flo.first_order_date1 = (SELECT MIN(first_order_date1) FROM FLO)
GROUP BY master_id, first_order_date1, DATEDIFF(day, first_order_date1, last_order_date1)
ORDER BY DateDif DESC
