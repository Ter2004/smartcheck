package com.smartcheck.app;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.core.app.NotificationCompat;
import androidx.core.content.ContextCompat;
import com.capacitorjs.plugins.pushnotifications.MessagingService;
import com.google.firebase.messaging.RemoteMessage;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Class-open notifications (app/services/class_push.py sends data-only messages).
 *
 * Before showing one, the phone listens for the room board for up to SCAN_MS:
 *   board heard     -> "you are in the room, tap to scan your face"
 *   board not heard -> "class has started, please hurry"
 *   cannot scan     -> the neutral text (Bluetooth off, no permission, blocked)
 * The server sends all three texts, so wording changes need no app update. The
 * tap still finds the board again on the check-in page: its code expires in 30 s.
 *
 * Replaces the push plugin's service (AndroidManifest.xml); token updates still
 * reach the plugin through super.onNewToken.
 */
public class ClassPushService extends MessagingService {

    private static final String TAG = "SmartCheckPush";
    static final String CHANNEL_ID = "checkin";          // same as push_notifications.js
    private static final long SCAN_MS = 4000;
    private static final int COMPANY_ID = 0xFFFF;         // firmware/smartcheck_broadcast
    private static final int ROOM_BYTES = 8;

    private enum Presence { IN_RANGE, OUT_OF_RANGE, UNKNOWN }

    @Override
    public void onMessageReceived(@NonNull RemoteMessage message) {
        Map<String, String> data = message.getData();
        if (!data.containsKey("session_id") || !data.containsKey("title_any")) {
            super.onMessageReceived(message);
            return;
        }
        Presence presence = scanForRoom(data.get("room"));
        Log.i(TAG, "kind=" + data.get("kind") + " presence=" + presence);
        String suffix = presence == Presence.IN_RANGE ? "_in" : presence == Presence.OUT_OF_RANGE ? "_out" : "_any";
        String title = data.containsKey("title" + suffix) ? data.get("title" + suffix) : data.get("title_any");
        String body = data.containsKey("body" + suffix) ? data.get("body" + suffix) : data.get("body_any");
        show(message, data, title, body);
    }

    /** Listen for the room board's advertisement ("SC", version 1, room code) for up to SCAN_MS. */
    @SuppressLint("MissingPermission")   // checked in canScan()
    private Presence scanForRoom(String room) {
        if (room == null || room.isEmpty() || room.length() > ROOM_BYTES || !canScan()) return Presence.UNKNOWN;
        BluetoothManager manager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        BluetoothAdapter adapter = manager == null ? null : manager.getAdapter();
        if (adapter == null || !adapter.isEnabled()) return Presence.UNKNOWN;
        BluetoothLeScanner scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) return Presence.UNKNOWN;

        // Filtered scan: Android stops unfiltered scans while the screen is off.
        byte[] prefix = new byte[3 + ROOM_BYTES];
        byte[] mask = new byte[prefix.length];
        prefix[0] = 'S';
        prefix[1] = 'C';
        prefix[2] = 1;
        byte[] roomBytes = room.getBytes(StandardCharsets.US_ASCII);
        System.arraycopy(roomBytes, 0, prefix, 3, roomBytes.length);   // rest stays 0x00 padding
        java.util.Arrays.fill(mask, (byte) 0xFF);
        ScanFilter filter = new ScanFilter.Builder().setManufacturerData(COMPANY_ID, prefix, mask).build();
        ScanSettings settings = new ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build();

        CountDownLatch found = new CountDownLatch(1);
        final boolean[] failed = {false};
        ScanCallback callback = new ScanCallback() {
            @Override
            public void onScanResult(int callbackType, ScanResult result) {
                found.countDown();
            }

            @Override
            public void onScanFailed(int errorCode) {
                Log.w(TAG, "scan failed: " + errorCode);
                failed[0] = true;
                found.countDown();
            }
        };
        try {
            scanner.startScan(Collections.singletonList(filter), settings, callback);
            boolean heard = found.await(SCAN_MS, TimeUnit.MILLISECONDS) && !failed[0];
            if (failed[0]) return Presence.UNKNOWN;
            return heard ? Presence.IN_RANGE : Presence.OUT_OF_RANGE;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return Presence.UNKNOWN;
        } catch (RuntimeException e) {   // SecurityException, adapter turned off meanwhile
            Log.w(TAG, "scan error: " + e.getClass().getSimpleName());
            return Presence.UNKNOWN;
        } finally {
            try {
                scanner.stopScan(callback);
            } catch (RuntimeException ignored) {
                // Bluetooth went off: the scan is already gone
            }
        }
    }

    /** The same permissions the check-in page asks for (bluetooth-le, androidNeverForLocation: false). */
    private boolean canScan() {
        if (!granted(Manifest.permission.ACCESS_FINE_LOCATION)) return false;
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.S || granted(Manifest.permission.BLUETOOTH_SCAN);
    }

    private boolean granted(String permission) {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED;
    }

    private void show(RemoteMessage message, Map<String, String> data, String title, String body) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && !granted(Manifest.permission.POST_NOTIFICATIONS)) return;
        NotificationManager manager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (manager == null) return;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && manager.getNotificationChannel(CHANNEL_ID) == null) {
            manager.createNotificationChannel(new NotificationChannel(CHANNEL_ID, "เช็คชื่อ", NotificationManager.IMPORTANCE_HIGH));
        }
        String sessionId = data.get("session_id");
        // The push plugin turns these extras into pushNotificationActionPerformed (needs
        // google.message_id); push_notifications.js then opens data.url.
        Intent open = new Intent(this, MainActivity.class)
            .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP)
            .putExtra("google.message_id", message.getMessageId() == null ? sessionId : message.getMessageId())
            .putExtra("session_id", sessionId)
            .putExtra("url", data.get("url"));
        PendingIntent tap = PendingIntent.getActivity(this, sessionId.hashCode(), open,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        NotificationCompat.Builder notification = new NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_checkin)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(new NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_REMINDER)
            .setAutoCancel(true)
            .setContentIntent(tap);
        // Same tag for both messages of a class: the reminder replaces the first one,
        // and the check-in page removes it by this tag (SmartCheckPush.clear).
        manager.notify(sessionId, 0, notification.build());
    }
}
