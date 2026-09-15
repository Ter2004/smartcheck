-- Only run in the disposable policy-test database. All fixtures roll back.
BEGIN;
DO $$
DECLARE admin_id uuid:=gen_random_uuid(); teacher_id uuid:=gen_random_uuid();
 other_teacher uuid:=gen_random_uuid(); student uuid:=gen_random_uuid(); outsider uuid:=gen_random_uuid();
 course uuid:=gen_random_uuid(); room uuid:=gen_random_uuid(); sid uuid:=gen_random_uuid();
 future_sid uuid:=gen_random_uuid(); sch uuid:=gen_random_uuid(); req uuid; blocked boolean;
BEGIN
 INSERT INTO users(id,email,full_name,role) VALUES
 (admin_id,admin_id||'@test.local','Admin','admin'),
 (teacher_id,teacher_id||'@test.local','Teacher','teacher'),
 (other_teacher,other_teacher||'@test.local','Other','teacher'),
 (student,student||'@test.local','Student','student'),
 (outsider,outsider||'@test.local','Outsider','student');
 INSERT INTO courses(id,code,name,teacher_id,semester) VALUES(course,'POLICY','Policy',teacher_id,1);
 INSERT INTO beacons(id,uuid,major,minor,room_name) VALUES(room,room::text,1,1,'Test room');
 INSERT INTO course_enrollments(course_id,student_id) VALUES(course,student);
 INSERT INTO consent_logs(user_id,consent_given) VALUES(student,true);
 INSERT INTO student_biometrics(user_id,face_embeddings,consent_given,enrolled_at)
 VALUES(student,'[[1,0]]',true,now());
 IF NOT (SELECT face_enrolled_once FROM users WHERE id=student) THEN RAISE EXCEPTION 'first enrollment not locked'; END IF;
 blocked:=false;
 BEGIN UPDATE student_biometrics SET face_embeddings='[[0,1]]' WHERE user_id=student;
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='face_replacement_not_approved' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'unapproved replacement allowed'; END IF;
 INSERT INTO face_change_requests(user_id,reason) VALUES(student,'Change requested') RETURNING id INTO req;
 blocked:=false;
 BEGIN UPDATE face_change_requests SET status='approved',reviewed_by=teacher_id,review_reason='Checked identity' WHERE id=req;
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='invalid_face_review' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'teacher approved face replacement'; END IF;
 UPDATE face_change_requests SET status='approved',reviewed_by=admin_id,review_reason='Checked official identity',reviewed_at=now() WHERE id=req;
 UPDATE student_biometrics SET face_embeddings='[[0,1]]' WHERE user_id=student;
 IF (SELECT status FROM face_change_requests WHERE id=req)!='consumed' THEN RAISE EXCEPTION 'approval not consumed'; END IF;
 blocked:=false;
 BEGIN UPDATE student_biometrics SET face_embeddings='[[1,1]]' WHERE user_id=student;
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='face_replacement_not_approved' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'approval reusable'; END IF;
 UPDATE student_biometrics SET face_embeddings=null,consent_given=false WHERE user_id=student;
 blocked:=false;
 BEGIN UPDATE student_biometrics SET face_embeddings='[[1,1]]',consent_given=true WHERE user_id=student;
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='face_replacement_not_approved' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'withdrawal bypassed enrollment lock'; END IF;

 INSERT INTO sessions(id,course_id,beacon_id,title,start_time,end_time,session_kind,created_by,change_reason,checkin_opens_at,late_at,checkin_closes_at)
 VALUES(sid,course,room,'Makeup',now()-interval '20 minutes',now()+interval '1 hour','makeup',admin_id,'Scheduled makeup',now()-interval '20 minutes',now()-interval '5 minutes',now()+interval '1 hour');
 INSERT INTO attendance(session_id,student_id,status,face_pass,ble_pass,liveness_pass)
 VALUES(sid,student,'present',true,true,true);
 IF (SELECT status FROM attendance WHERE session_id=sid AND student_id=student)!='late' THEN RAISE EXCEPTION 'late not enforced'; END IF;
 blocked:=false;
 BEGIN UPDATE attendance SET override_by=other_teacher,override_reason='Checked attendance',status='present' WHERE session_id=sid;
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='invalid_override' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'cross teacher override allowed'; END IF;
 blocked:=false;
 BEGIN INSERT INTO attendance(session_id,student_id,status,override_by,override_reason) VALUES(sid,outsider,'manual',teacher_id,'Checked attendance');
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='student_not_in_roster' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'outside roster override allowed'; END IF;
 UPDATE attendance SET override_by=teacher_id,override_reason='Checked attendance',status='present' WHERE session_id=sid;
 IF (SELECT count(*) FROM audit_logs WHERE session_id=sid AND event_type='teacher_override')!=1 THEN RAISE EXCEPTION 'audit missing or rejected override audited'; END IF;
 blocked:=false;
 BEGIN UPDATE attendance SET override_reason='' WHERE session_id=sid;
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='invalid_override' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'empty reason allowed'; END IF;

 INSERT INTO schedules(id,course_id,day_of_week,start_time,end_time,beacon_id) VALUES(sch,course,1,'09:00','12:00',room);
 INSERT INTO sessions(id,course_id,beacon_id,schedule_id,title,start_time,end_time,session_kind,checkin_opens_at,late_at,checkin_closes_at)
 VALUES(future_sid,course,room,sch,'Future',now()+interval '1 day',now()+interval '25 hours','scheduled',now()+interval '1 day',now()+interval '24 hours 15 minutes',now()+interval '25 hours');
 blocked:=false;
 BEGIN INSERT INTO attendance(session_id,student_id,status) VALUES(future_sid,student,'present');
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='session_outside_checkin_window' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'future checkin allowed'; END IF;
 UPDATE schedules SET is_active=false,retired_by=admin_id,retire_reason='Cancelled future meetings' WHERE id=sch;
 IF (SELECT cancelled_at FROM sessions WHERE id=future_sid) IS NULL THEN RAISE EXCEPTION 'future occurrence not cancelled'; END IF;
 IF (SELECT cancelled_at FROM sessions WHERE id=sid) IS NOT NULL THEN RAISE EXCEPTION 'unrelated session cancelled'; END IF;
 blocked:=false;
 BEGIN UPDATE sessions SET cancelled_at=null WHERE id=future_sid;
 EXCEPTION WHEN OTHERS THEN IF SQLERRM='cannot_reopen_cancelled_session' THEN blocked:=true; ELSE RAISE; END IF; END;
 IF NOT blocked THEN RAISE EXCEPTION 'cancelled session reopened'; END IF;
 RAISE NOTICE 'PASS: enrollment lock, one-use approval, withdrawal, authority, roster, late, audit, time gate, schedule cancellation';
END $$;
ROLLBACK;
