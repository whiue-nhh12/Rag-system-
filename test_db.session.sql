BEGIN;
SELECT * FROM usertest;
SELECT * FROM usertest;
COMMIT;

BEGIN;
INSERT INTO usertest(id,fullname) VALUES (9,'Hieu');
INSERT INTO usertest(id,fullname) VALUES (10,'Hieu1');
COMMIT;

